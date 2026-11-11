# okc: problem=muse_moe_w4a16
"""Probe M8, not a submission: per-stage graph-replay times of the Triton MoE (align, GEMM1, GEMM2, finalize) at
its best tiles, to see where it loses to Marlin. Was probe M7: Triton NVFP4 MoE with the PTX dequant + nibble repack (contiguous activation halves)
+ per-block scales, vs SGLang's Marlin path per batch size, correctness against Marlin, with a tile sweep."""

import math
import time
from typing import TypedDict

import torch
import triton
import triton.language as tl
import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


@triton.jit
def _e2m1_scaled(code, sf):
    """E2M1 code (int32, 4 bits) times its block scale (fp32) -> bf16, exactly."""
    mag = code & 7
    # mag >= 2: exponent (mag >> 1) - 1, mantissa (mag & 1) / 2, i.e. FP32 bits (mag + 252) << 22.
    big = ((mag + 252) << 22).to(tl.float32, bitcast=True)
    v = tl.where(mag >= 2, big, mag.to(tl.float32) * 0.5)
    v = tl.where((code & 8) != 0, -v, v)
    return (v * sf).to(tl.bfloat16)


_ASM_LO = """
{
.reg .b32 t, a, b, z, bias, nz;
mov.b32 z, 0;
mov.b32 bias, 0x7E807E80;
mov.b32 nz, 0x80008000;
prmt.b32 t, $2, z, 0x4140;
and.b32 a, t, 0x00070007;
shl.b32 a, a, 6;
and.b32 b, t, 0x00080008;
shl.b32 b, b, 12;
or.b32 a, a, b;
fma.rn.bf16x2 a, a, bias, nz;
fma.rn.bf16x2 $0, a, $3, nz;
prmt.b32 t, $2, z, 0x4342;
and.b32 a, t, 0x00070007;
shl.b32 a, a, 6;
and.b32 b, t, 0x00080008;
shl.b32 b, b, 12;
or.b32 a, a, b;
fma.rn.bf16x2 a, a, bias, nz;
fma.rn.bf16x2 $1, a, $4, nz;
}
"""
_ASM_HI = tl.constexpr(_ASM_LO.replace("0x00070007", "0x00700070").replace("shl.b32 a, a, 6;", "shl.b32 a, a, 2;")
                       .replace("0x00080008", "0x00800080").replace("shl.b32 b, b, 12;", "shl.b32 b, b, 8;"))
_ASM_LO = tl.constexpr(_ASM_LO)  # kernels may only read constexpr globals


@triton.jit
def _halves_scales(s_rows, kk, j16, BN: tl.constexpr, BK: tl.constexpr):
    """The bf16 block scales of K = kk .. kk + BK/2 - 1, loaded once per 16 values, broadcast to [BN, BK/2]."""
    sf = tl.load(s_rows + kk // 16 + j16[None, :]).to(tl.bfloat16)  # [BN, BK/32]
    return tl.reshape(tl.broadcast_to(sf[:, :, None], [BN, BK // 32, 16]), [BN, BK // 2])


@triton.jit
def _fp4_planes(p, sf_lo, sf_hi, FAST: tl.constexpr):
    """Packed E2M1 bytes p (uint8) and the scales of each plane (bf16) -> (low-nibble values, high-nibble values),
    both bf16 and exact."""
    if FAST:
        lo = tl.inline_asm_elementwise(_ASM_LO, "=r,=r,r,r,r", [p, sf_lo], dtype=tl.bfloat16, is_pure=True, pack=4)
        hi = tl.inline_asm_elementwise(_ASM_HI, "=r,=r,r,r,r", [p, sf_hi], dtype=tl.bfloat16, is_pure=True, pack=4)
    else:
        q = p.to(tl.int32)
        lo = _e2m1_scaled(q & 15, sf_lo.to(tl.float32))
        hi = _e2m1_scaled(q >> 4, sf_hi.to(tl.float32))
    return lo, hi


@triton.jit
def _align_kernel(ids_ptr, sorted_ptr, block_expert_ptr, local_start, n_slots, max_blocks,
                  COUNT: tl.constexpr, BINS: tl.constexpr, BM: tl.constexpr, NS: tl.constexpr, NB: tl.constexpr):
    """One program: the slots routed to local experts, grouped by expert into BM-row blocks.
    sorted[b * BM + r] = slot (token * 8 + k) or -1; block_expert[b] = local expert or -1."""
    s = tl.arange(0, NS)
    s_ok = s < n_slots
    e = tl.load(ids_ptr + s, mask=s_ok, other=-1) - local_start
    local = s_ok & (e >= 0) & (e < COUNT)
    e = tl.where(local, e, COUNT)  # non-local last
    counts = tl.histogram(e, BINS)  # BINS: a power of two > COUNT
    ex = tl.arange(0, BINS)
    counts = tl.where(ex < COUNT, counts, 0)
    blocks = (counts + BM - 1) // BM
    block_end = tl.cumsum(blocks, 0)
    block_start = block_end - blocks
    # Sort slots by (expert, slot); rank within the expert = sorted position - expert's first position.
    key = e * NS + s
    key_sorted = tl.sort(key)
    e_sorted = key_sorted // NS
    s_sorted = key_sorted % NS
    first = tl.cumsum(counts, 0) - counts  # sorted position of each expert's first slot
    pos = tl.arange(0, NS)
    # rank = pos - first[e_sorted]; dest row = block_start[e_sorted] * BM + rank
    first_g = tl.gather(first, e_sorted, 0)
    bstart_g = tl.gather(block_start, e_sorted, 0)
    dest = bstart_g * BM + (pos - first_g)
    keep = (e_sorted < COUNT) & (s_sorted < n_slots)
    # Pad rows first (-1), then the real slots.
    rows = tl.arange(0, NB * BM)
    tl.store(sorted_ptr + rows, tl.full([NB * BM], -1, tl.int32), mask=rows < max_blocks * BM)
    tl.debug_barrier()
    tl.store(sorted_ptr + dest, s_sorted.to(tl.int32), mask=keep)
    b = tl.arange(0, NB)
    be = tl.sum(tl.where((b[:, None] >= block_start[None, :]) & (b[:, None] < block_end[None, :]),
                         ex[None, :], 0), 1)
    used = b < tl.max(block_end, 0)
    tl.store(block_expert_ptr + b, tl.where(used, be, -1).to(tl.int32), mask=b < max_blocks)


@triton.jit
def _gemm1_kernel(x_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, inter_ptr,
                  H, I, TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FAST: tl.constexpr):
    """inter[slot, n] = silu(x[tok] @ gate[e, n]^T) * (x[tok] @ up[e, n]^T) for BM slots x BN columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    tok = tl.where(row_ok, slot // TOPK, 0)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    e64 = e.to(tl.int64)
    wg = w_ptr + (e64 * (2 * I) + n)[:, None] * (H // 2)          # gate rows
    wu = w_ptr + (e64 * (2 * I) + I + n)[:, None] * (H // 2)      # up rows
    sg = sf_ptr + (e64 * (2 * I) + n)[:, None] * (H // 16)
    su = sf_ptr + (e64 * (2 * I) + I + n)[:, None] * (H // 16)
    xrow = x_ptr + tok[:, None].to(tl.int64) * H
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    j16 = tl.arange(0, BK // 32)
    for k0 in range(0, H, BK):
        xa = tl.load(xrow + k0 + i[None, :], mask=row_ok[:, None], other=0.0)
        xb = tl.load(xrow + k0 + BK // 2 + i[None, :], mask=row_ok[:, None], other=0.0)
        pg = tl.load(wg + k0 // 2 + i[None, :])
        pu = tl.load(wu + k0 // 2 + i[None, :])
        g_lo, g_hi = _fp4_planes(pg, _halves_scales(sg, k0, j16, BN, BK), _halves_scales(sg, k0 + BK // 2, j16, BN, BK), FAST)
        u_lo, u_hi = _fp4_planes(pu, _halves_scales(su, k0, j16, BN, BK), _halves_scales(su, k0 + BK // 2, j16, BN, BK), FAST)
        acc_g = tl.dot(xa, tl.trans(g_lo), acc_g)
        acc_g = tl.dot(xb, tl.trans(g_hi), acc_g)
        acc_u = tl.dot(xa, tl.trans(u_lo), acc_u)
        acc_u = tl.dot(xb, tl.trans(u_hi), acc_u)
    gs = tl.load(gs_ptr + e)
    g = acc_g * gs
    u = acc_u * gs
    h = g * tl.sigmoid(g) * u
    dst = inter_ptr + tl.where(row_ok, slot, 0)[:, None].to(tl.int64) * I + n[None, :]
    tl.store(dst, h.to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit
def _gemm2_kernel(inter_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, y_ptr,
                  H, I, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FAST: tl.constexpr):
    """y[slot, n] = inter[slot] @ w2[e, n]^T for BM slots x BN columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    srow = tl.where(row_ok, slot, 0).to(tl.int64)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    e64 = e.to(tl.int64)
    wp = w_ptr + (e64 * H + n)[:, None] * (I // 2)
    sp = sf_ptr + (e64 * H + n)[:, None] * (I // 16)
    arow = inter_ptr + srow[:, None] * I
    acc = tl.zeros([BM, BN], tl.float32)
    j16 = tl.arange(0, BK // 32)
    for k0 in range(0, I, BK):
        aa = tl.load(arow + k0 + i[None, :], mask=row_ok[:, None], other=0.0)
        ab = tl.load(arow + k0 + BK // 2 + i[None, :], mask=row_ok[:, None], other=0.0)
        p = tl.load(wp + k0 // 2 + i[None, :])
        w_lo, w_hi = _fp4_planes(p, _halves_scales(sp, k0, j16, BN, BK), _halves_scales(sp, k0 + BK // 2, j16, BN, BK), FAST)
        acc = tl.dot(aa, tl.trans(w_lo), acc)
        acc = tl.dot(ab, tl.trans(w_hi), acc)
    y = acc * tl.load(gs_ptr + e)
    tl.store(y_ptr + srow[:, None] * H + n[None, :], y.to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit
def _finalize_kernel(y_ptr, ids_ptr, wts_ptr, norm_w_ptr, out_ptr, local_start, H,
                     COUNT: tl.constexpr, TOPK: tl.constexpr, BH: tl.constexpr, EPS: tl.constexpr):
    """out[t] = sum over t's local slots of weight * rmsnorm(y * (1 + w))."""
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    h_ok = h < H
    w1 = 1.0 + tl.load(norm_w_ptr + h, mask=h_ok, other=0.0).to(tl.float32)
    out = tl.zeros([BH], tl.float32)
    for k in range(TOPK):
        e = tl.load(ids_ptr + t * TOPK + k) - local_start
        if (e >= 0) & (e < COUNT):
            z = tl.load(y_ptr + (t * TOPK + k).to(tl.int64) * H + h, mask=h_ok, other=0.0).to(tl.float32) * w1
            inv = tl.rsqrt(tl.sum(z * z, 0) / H + EPS)
            out += tl.load(wts_ptr + t * TOPK + k) * z * inv
    tl.store(out_ptr + t * H + h, out.to(tl.bfloat16), mask=h_ok)


def repack(packed: torch.Tensor, bk: int) -> torch.Tensor:
    """[..., K/2] uint8 (byte i = K 2i low nibble, 2i+1 high) -> the same codes with, in each bk-wide K chunk, byte j
    = K j (low nibble) and K bk/2 + j (high nibble). A permutation: lossless."""
    lo, hi = packed & 15, packed >> 4
    codes = torch.stack((lo, hi), -1).flatten(-2)  # [..., K]
    k = codes.shape[-1]
    c = codes.view(*codes.shape[:-1], k // bk, 2, bk // 2)
    return (c[..., 0, :] | (c[..., 1, :] << 4)).reshape(*codes.shape[:-1], k // 2).contiguous()


def block_m(m: int) -> int:
    """Rows per expert block from the token count (shape-only)."""
    return 16 if m <= 16 else 32 if m <= 64 else 64


def moe_forward(x, topk_ids, topk_wts, w13, w13_sf, w13_gs, w2, w2_sf, w2_gs, norm_w, local_start: int,
                count: int, topk: int = 8, bn1: int = 64, bn2: int = 64, bk: int = 64, warps: int = 4,
                stages: int = 3, eps: float = 1e-8, bm: int = 0, fast: bool = True):
    """w13 [count, 2I, H/2] uint8, w13_sf [count, 2I, H/16] e4m3, w13_gs [count] fp32 (the local experts);
    w2 [count, H, I/2], w2_sf [count, H, I/16], w2_gs [count]. Returns bf16 [M, H]."""
    m, h = x.shape
    i_dim = w2.shape[2] * 2
    bm = bm or block_m(m)
    n_slots = m * topk
    max_blocks = triton.cdiv(n_slots, bm) + count
    dev = x.device
    sorted_ids = torch.empty((max_blocks * bm,), dtype=torch.int32, device=dev)
    block_expert = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
    _align_kernel[(1,)](topk_ids, sorted_ids, block_expert, local_start, n_slots, max_blocks,
                        COUNT=count, BINS=triton.next_power_of_2(count + 1), BM=bm,
                        NS=max(16, triton.next_power_of_2(n_slots)),
                        NB=triton.next_power_of_2(max_blocks), num_warps=4)
    inter = torch.empty((n_slots, i_dim), dtype=torch.bfloat16, device=dev)
    _gemm1_kernel[(max_blocks, i_dim // bn1)](x, sorted_ids, block_expert, w13, w13_sf, w13_gs, inter, h, i_dim,
                                              TOPK=topk, BM=bm, BN=bn1, BK=bk, FAST=fast, num_warps=warps,
                                              num_stages=stages)
    y = torch.empty((n_slots, h), dtype=torch.bfloat16, device=dev)
    _gemm2_kernel[(max_blocks, h // bn2)](inter, sorted_ids, block_expert, w2, w2_sf, w2_gs, y, h, i_dim,
                                          BM=bm, BN=bn2, BK=bk, FAST=fast, num_warps=warps,
                                          num_stages=stages)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    _finalize_kernel[(m,)](y, topk_ids, topk_wts, norm_w, out, local_start, h, COUNT=count, TOPK=topk,
                           BH=triton.next_power_of_2(h), EPS=eps, num_warps=8)
    return out


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def marlin_prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def marlin_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _inputs(bs, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.25 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def prepare(weights):
    t0 = time.time()
    state = marlin_prepare(weights)
    dev = weights["w13"].device
    W = {k: weights[k][:32] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    REP = {bk: (repack(W["w13"], bk), repack(W["w2"], bk)) for bk in (64, 128)}
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__} prep {time.time() - t0:.0f}s"]

    def bench(fn, reps=20):
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        ts = []
        for _ in range(reps):
            torch.sum(flush, dim=0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gr.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    nop = torch.zeros(1, device=dev)
    lines.append(f"nop {bench(lambda: nop.add_(1)):.1f}")
    best = {1: (64, 64, 128, 8, 4, 16), 8: (64, 64, 128, 8, 4, 16), 32: (64, 64, 128, 4, 3, 16), 256: (64, 128, 128, 4, 3, 64)}
    for bs, (bn1, bn2, bk, warps, stages, bm) in best.items():
        data = _inputs(bs, 7, dev)
        x, ids, wts = data["hidden_states"], data["topk_ids"], data["topk_weights"]
        m, h = x.shape
        i_dim = 4096
        n_slots = m * 8
        max_blocks = triton.cdiv(n_slots, bm) + 32
        sorted_ids = torch.empty((max_blocks * bm,), dtype=torch.int32, device=dev)
        block_expert = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
        inter = torch.empty((n_slots, i_dim), dtype=torch.bfloat16, device=dev)
        y = torch.empty((n_slots, h), dtype=torch.bfloat16, device=dev)
        out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
        w13, w2 = REP[bk]
        al = lambda: _align_kernel[(1,)](ids, sorted_ids, block_expert, 0, n_slots, max_blocks, COUNT=32, BINS=64, BM=bm,
                                         NS=max(16, triton.next_power_of_2(n_slots)), NB=triton.next_power_of_2(max_blocks), num_warps=4)
        g1 = lambda: _gemm1_kernel[(max_blocks, i_dim // bn1)](x, sorted_ids, block_expert, w13, W["w13_sf"], W["w13_gs"], inter, h,
                                                                i_dim, TOPK=8, BM=bm, BN=bn1, BK=bk, FAST=True, num_warps=warps,
                                                                num_stages=stages)
        g2 = lambda: _gemm2_kernel[(max_blocks, h // bn2)](inter, sorted_ids, block_expert, w2, W["w2_sf"], W["w2_gs"], y, h, i_dim,
                                                            BM=bm, BN=bn2, BK=bk, FAST=True, num_warps=warps, num_stages=stages)
        fi = lambda: _finalize_kernel[(m,)](y, ids, wts, weights["post_norm_w"], out, 0, h, COUNT=32, TOPK=8,
                                            BH=triton.next_power_of_2(h), EPS=1e-8, num_warps=8)
        for f in (al, g1, g2, fi):
            f()
        torch.cuda.synchronize()
        used = int((block_expert >= 0).sum())
        tok = int(((ids >= 0) & (ids < 32)).sum())
        gb1 = used / max(1, used) * int(torch.unique(ids[ids < 32]).numel()) * (8192 * 2048 + 8192 * 256) / 1e9
        t = [bench(f) for f in (al, g1, g2, fi)]
        lines.append(f"bs{bs} blocks_used {used} local_routes {tok} | align {t[0]:.1f} gemm1 {t[1]:.1f} gemm2 {t[2]:.1f} "
                     f"finalize {t[3]:.1f} us (nop-included) | gemm1 weight GB {gb1:.3f} -> {gb1 / max(1e-9, (t[1] - 5.2) * 1e-6) / 1e3:.2f} TB/s")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
