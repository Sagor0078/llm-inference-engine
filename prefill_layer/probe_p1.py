# okc: problem=muse_prefill_layer
"""Probe P1, not a submission: the MoE experts at prefill sizes (753 / 3012 / 16384 tokens, ~6K-131K routed slots):
SGLang's Marlin expert path (as the layer runs it) vs the tile-major Triton grouped GEMM (moe_triton5) with large tiles,
on this layer's expert weights. CUDA-graph replays from a cold L2. Runs in prepare(), then stops the run."""

import math
import time

import triton
import triton.language as tl
from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl



HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
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
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)




# ---- moe_triton5 ----
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
def _wtile(w_ptr, sf_ptr, row0, nrows: tl.constexpr, kdim: tl.constexpr, c, i, j16, BN: tl.constexpr,
           BK: tl.constexpr, TILED: tl.constexpr, FAST: tl.constexpr):
    """Rows row0 .. row0 + BN - 1 (of one expert's [nrows, kdim] matrix, the pointers already at that expert), K-chunk
    c: the two dequantized planes (bf16 [BN, BK/2] each: K = c*BK + j and c*BK + BK/2 + j)."""
    r = tl.arange(0, BN)
    if TILED:
        tile = (row0 // BN) * (kdim // BK) + c
        wp = w_ptr + tile.to(tl.int64) * (BN * (BK // 2)) + r[:, None] * (BK // 2) + i[None, :]
        sp = sf_ptr + tile.to(tl.int64) * (BN * (BK // 16)) + r[:, None] * (BK // 16)
    else:
        wp = w_ptr + (row0 + r)[:, None].to(tl.int64) * (kdim // 2) + c * (BK // 2) + i[None, :]
        sp = sf_ptr + (row0 + r)[:, None].to(tl.int64) * (kdim // 16) + c * (BK // 16)
    p = tl.load(wp)
    s_lo = tl.load(sp + j16[None, :]).to(tl.bfloat16)               # [BN, BK/32]: first half's blocks
    s_hi = tl.load(sp + BK // 32 + j16[None, :]).to(tl.bfloat16)    # second half's
    s_lo = tl.reshape(tl.broadcast_to(s_lo[:, :, None], [BN, BK // 32, 16]), [BN, BK // 2])
    s_hi = tl.reshape(tl.broadcast_to(s_hi[:, :, None], [BN, BK // 32, 16]), [BN, BK // 2])
    return _fp4_planes(p, s_lo, s_hi, FAST)


@triton.jit(do_not_specialize=["SPLIT"])
def _gemm1_kernel(x_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, inter_ptr, wsg_ptr, wsu_ptr, cnt_ptr,
                  SPLIT, H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, FAST: tl.constexpr, TILED: tl.constexpr):
    """inter[slot, n] = silu(x[tok] @ gate[e, n]^T) * (x[tok] @ up[e, n]^T) for BM slots x BN columns, over this
    unit's 1/SPLIT of K. With SPLIT > 1 the partials meet in zeroed FP32 workspaces and the last unit to finish a
    tile applies SiLU x up and re-zeroes them (the split-K hand-off proven on muse_bf16_proj)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    tok = tl.where(row_ok, slot // TOPK, 0)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    j16 = tl.arange(0, BK // 32)
    e64 = e.to(tl.int64)
    we = w_ptr + e64 * (2 * I) * (H // 2)       # this expert's [2I, H/2] codes (gate rows, then up rows)
    se = sf_ptr + e64 * (2 * I) * (H // 16)
    xrow = x_ptr + tok[:, None].to(tl.int64) * H
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    chunks = H // BK
    c0 = pid_k * chunks // SPLIT
    c1 = (pid_k + 1) * chunks // SPLIT
    for c in range(c0, c1):
        k0 = c * BK
        xa = tl.load(xrow + k0 + i[None, :], mask=row_ok[:, None], other=0.0)
        xb = tl.load(xrow + k0 + BK // 2 + i[None, :], mask=row_ok[:, None], other=0.0)
        g_lo, g_hi = _wtile(we, se, pid_n * BN, 2 * I, H, c, i, j16, BN, BK, TILED, FAST)
        u_lo, u_hi = _wtile(we, se, I + pid_n * BN, 2 * I, H, c, i, j16, BN, BK, TILED, FAST)
        acc_g = tl.dot(xa, tl.trans(g_lo), acc_g)
        acc_g = tl.dot(xb, tl.trans(g_hi), acc_g)
        acc_u = tl.dot(xa, tl.trans(u_lo), acc_u)
        acc_u = tl.dot(xb, tl.trans(u_hi), acc_u)
    srow = tl.where(row_ok, slot, 0)[:, None].to(tl.int64)
    if SPLIT > 1:
        wsg = wsg_ptr + srow * I + n[None, :]
        wsu = wsu_ptr + srow * I + n[None, :]
        tl.atomic_add(wsg, acc_g, mask=row_ok[:, None], sem="relaxed")
        tl.atomic_add(wsu, acc_u, mask=row_ok[:, None], sem="relaxed")
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(1) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT - 1:
            acc_g = tl.load(wsg, mask=row_ok[:, None], other=0.0, cache_modifier=".cg")
            acc_u = tl.load(wsu, mask=row_ok[:, None], other=0.0, cache_modifier=".cg")
            gs = tl.load(gs_ptr + e)
            g = acc_g * gs
            h = g * tl.sigmoid(g) * (acc_u * gs)
            tl.store(inter_ptr + srow * I + n[None, :], h.to(tl.bfloat16), mask=row_ok[:, None])
            tl.debug_barrier()  # every thread has read before any re-zeroes
            tl.store(wsg, tl.zeros([BM, BN], tl.float32), mask=row_ok[:, None])
            tl.store(wsu, tl.zeros([BM, BN], tl.float32), mask=row_ok[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")
    else:
        gs = tl.load(gs_ptr + e)
        g = acc_g * gs
        h = g * tl.sigmoid(g) * (acc_u * gs)
        tl.store(inter_ptr + srow * I + n[None, :], h.to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit(do_not_specialize=["SPLIT"])
def _gemm2_kernel(inter_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, y_ptr, wsy_ptr, cnt_ptr,
                  SPLIT, H: tl.constexpr, I: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FAST: tl.constexpr, TILED: tl.constexpr):
    """y[slot, n] = inter[slot] @ w2[e, n]^T for BM slots x BN columns, split-K as GEMM1."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    srow = tl.where(row_ok, slot, 0).to(tl.int64)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    j16 = tl.arange(0, BK // 32)
    e64 = e.to(tl.int64)
    we = w_ptr + e64 * H * (I // 2)
    se = sf_ptr + e64 * H * (I // 16)
    arow = inter_ptr + srow[:, None] * I
    acc = tl.zeros([BM, BN], tl.float32)
    chunks = I // BK
    c0 = pid_k * chunks // SPLIT
    c1 = (pid_k + 1) * chunks // SPLIT
    for c in range(c0, c1):
        k0 = c * BK
        aa = tl.load(arow + k0 + i[None, :], mask=row_ok[:, None], other=0.0)
        ab = tl.load(arow + k0 + BK // 2 + i[None, :], mask=row_ok[:, None], other=0.0)
        w_lo, w_hi = _wtile(we, se, pid_n * BN, H, I, c, i, j16, BN, BK, TILED, FAST)
        acc = tl.dot(aa, tl.trans(w_lo), acc)
        acc = tl.dot(ab, tl.trans(w_hi), acc)
    dst = y_ptr + srow[:, None] * H + n[None, :]
    if SPLIT > 1:
        ws = wsy_ptr + srow[:, None] * H + n[None, :]
        tl.atomic_add(ws, acc, mask=row_ok[:, None], sem="relaxed")
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(1) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT - 1:
            total = tl.load(ws, mask=row_ok[:, None], other=0.0, cache_modifier=".cg")
            tl.store(dst, (total * tl.load(gs_ptr + e)).to(tl.bfloat16), mask=row_ok[:, None])
            tl.debug_barrier()  # every thread has read before any re-zeroes
            tl.store(ws, tl.zeros([BM, BN], tl.float32), mask=row_ok[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")
    else:
        tl.store(dst, (acc * tl.load(gs_ptr + e)).to(tl.bfloat16), mask=row_ok[:, None])


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


def tile_major(t: torch.Tensor, bn: int, bk: int, per: int) -> torch.Tensor:
    """[E, N, K/per] -> [E, N/bn, K/bk, bn, bk/per]: the same bytes, tile-major (lossless). per = 2 for packed
    codes (after repack(..., bk)), 16 for block scales."""
    e, n, kp = t.shape
    return t.view(e, n // bn, bn, kp // (bk // per), bk // per).permute(0, 1, 3, 2, 4).contiguous()


def block_m(m: int) -> int:
    """Rows per expert block from the token count (shape-only)."""
    return 16 if m <= 16 else 32 if m <= 64 else 64


def make_workspace(max_slots: int, h: int, i_dim: int, device) -> dict:
    """Zeroed split-K workspaces (re-zeroed by the kernels after every use) for up to max_slots routed slots."""
    return {"g": torch.zeros((max_slots * i_dim,), dtype=torch.float32, device=device),
            "u": torch.zeros((max_slots * i_dim,), dtype=torch.float32, device=device),
            "y": torch.zeros((max_slots * h,), dtype=torch.float32, device=device),
            "cnt1": torch.zeros((1 << 16,), dtype=torch.int32, device=device),
            "cnt2": torch.zeros((1 << 16,), dtype=torch.int32, device=device), "max_slots": max_slots}


def moe_forward(x, topk_ids, topk_wts, w13, w13_sf, w13_gs, w2, w2_sf, w2_gs, norm_w, local_start: int,
                count: int, topk: int = 8, bn1: int = 64, bn2: int = 64, bk: int = 64, warps: int = 4,
                stages: int = 3, eps: float = 1e-8, bm: int = 0, fast: bool = True, split1: int = 1, split2: int = 1,
                ws=None, tiled: bool = False):
    """w13 / w2 repacked with repack(..., bk). split1 / split2 > 1 need ws (make_workspace) covering m * topk slots.
    Returns bf16 [M, H]."""
    m, h = x.shape
    i_dim = w2.shape[-1] * w2.shape[-3] * 2 if tiled else w2.shape[2] * 2
    bm = bm or block_m(m)
    n_slots = m * topk
    max_blocks = triton.cdiv(n_slots, bm) + count
    dev = x.device
    if split1 > 1 or split2 > 1:
        assert ws is not None and n_slots <= ws["max_slots"]
        assert max_blocks * (max(i_dim // bn1, h // bn2)) <= ws["cnt1"].numel()
    dummy = torch.empty((1,), dtype=torch.float32, device=dev)
    sorted_ids = torch.empty((max_blocks * bm,), dtype=torch.int32, device=dev)
    block_expert = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
    _align_kernel[(1,)](topk_ids, sorted_ids, block_expert, local_start, n_slots, max_blocks,
                        COUNT=count, BINS=triton.next_power_of_2(count + 1), BM=bm,
                        NS=max(16, triton.next_power_of_2(n_slots)), NB=triton.next_power_of_2(max_blocks), num_warps=4)
    inter = torch.empty((n_slots, i_dim), dtype=torch.bfloat16, device=dev)
    _gemm1_kernel[(max_blocks, i_dim // bn1, split1)](
        x, sorted_ids, block_expert, w13, w13_sf, w13_gs, inter, ws["g"] if ws else dummy, ws["u"] if ws else dummy,
        ws["cnt1"] if ws else dummy, split1, H=h, I=i_dim, TOPK=topk, BM=bm, BN=bn1, BK=bk, FAST=fast, TILED=tiled, num_warps=warps,
        num_stages=stages)
    y = torch.empty((n_slots, h), dtype=torch.bfloat16, device=dev)
    _gemm2_kernel[(max_blocks, h // bn2, split2)](
        inter, sorted_ids, block_expert, w2, w2_sf, w2_gs, y, ws["y"] if ws else dummy, ws["cnt2"] if ws else dummy,
        split2, H=h, I=i_dim, BM=bm, BN=bn2, BK=bk, FAST=fast, TILED=tiled, num_warps=warps, num_stages=stages)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    _finalize_kernel[(m,)](y, topk_ids, topk_wts, norm_w, out, local_start, h, COUNT=count, TOPK=topk,
                           BH=triton.next_power_of_2(h), EPS=eps, num_warps=8)
    return out



def _route(tokens, seed, dev):
    """Skewed top-8 routing like the layer's (a few hot experts), with this GPU's experts 0-31 the busiest."""
    g = torch.Generator(device=dev).manual_seed(seed)
    scores = torch.rand((tokens, 256), generator=g, device=dev) + torch.rand((256,), generator=g, device=dev).pow(6) * 4
    routed = scores.topk(8, -1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = ((ids - 32 * pairs.argmax()) % 256).to(torch.int32)
    return ids, (routed.values / routed.values.sum(-1, keepdim=True)).float()


def prepare(weights):
    t0 = time.time()
    state = {"bank": _marlin_bank(weights),
             "post_expert_w": (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16)}
    dev = weights["w13"].device
    lines = []
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)

    def bench(fn, reps=5):
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

    REP = {}
    for bn, bk in ((128, 64), (128, 128), (64, 128)):
        r13, r2 = repack(weights["w13"], bk), repack(weights["w2"], bk)
        REP[(bn, bk)] = (tile_major(r13, bn, bk, 2), tile_major(weights["w13_sf"], bn, bk, 16),
                         tile_major(r2, bn, bk, 2), tile_major(weights["w2_sf"], bn, bk, 16))
        del r13, r2
    lines.append(f"repack [{time.time() - t0:.0f}s]")
    gs13 = weights["w13_gs"].float().contiguous()
    gs2 = weights["w2_gs"].float().contiguous()
    for tokens in (753, 3012, 16384):
        if time.time() - t0 > 250:
            lines.append("TIME BUDGET HIT")
            break
        x = torch.randn((tokens, 4096), device=dev).to(torch.bfloat16)
        ids, wts = _route(tokens, tokens, dev)
        local = int(((ids >= 0) & (ids < 32)).sum())
        marlin = lambda: _experts(x, wts, ids, state)
        ref = marlin().float()
        tm = bench(marlin)
        res = []
        for (bn, bk), (w13, s13, w2, s2) in REP.items():
            for bm in (64, 128):
                for warps in (4, 8):
                    for stages in (2, 3):
                        if time.time() - t0 > 250 or (bm * bn >= 16384 and warps == 4):
                            continue
                        tag = f"bm{bm},bn{bn},bk{bk},w{warps},st{stages}"
                        fn = lambda w13=w13, s13=s13, w2=w2, s2=s2, bn=bn, bk=bk, bm=bm, warps=warps, stages=stages: moe_forward(
                            x, ids, wts, w13, s13, gs13, w2, s2, gs2, weights["post_expert_norm"], 0, 32, bn1=bn, bn2=bn,
                            bk=bk, warps=warps, stages=stages, bm=bm, fast=True, tiled=True)
                        try:
                            out = fn().float()
                            err = float((out - ref).norm() / ref.norm())
                            if not math.isfinite(err) or err > 0.02:
                                res.append((1e9, tag + f"BAD{err:.1e}"))
                                continue
                            res.append((bench(fn), tag))
                        except Exception as e:  # noqa: BLE001
                            res.append((1e9, tag + type(e).__name__[:12] + str(e)[:60]))
        res.sort()
        bad = [t for v, t in res if v >= 1e9]
        lines.append(f"tokens {tokens} local_routes {local} marlin {tm:.1f} | triton " + " ".join(f"{v:.1f}/{t}" for v, t in res[:4])
                     + f" | bad {len(bad)} {bad[0] if bad else ''} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
        torch.cuda.empty_cache()
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
