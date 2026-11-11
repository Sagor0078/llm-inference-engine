# okc: problem=muse_moe_w4a16
"""Probe M10, not a submission: tile-major vs row-major split-K Triton MoE vs the v2 submission path.
Was probe M7: Triton NVFP4 MoE with the PTX dequant + nibble repack (contiguous activation halves)
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


@triton.jit
def _v2_finalize_kernel(y_ptr, ids_ptr, wts_ptr, norm_w_ptr, out_ptr, local_start, H,
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


@triton.jit
def _v2_align_kernel(ids_ptr, sorted_ptr, block_expert_ptr, npad_ptr, local_ptr, local_start, n_slots, max_blocks,
                  COUNT: tl.constexpr, BINS: tl.constexpr, BM: tl.constexpr, NS: tl.constexpr, NB: tl.constexpr,
                  ROWS: tl.constexpr):
    """One program, in Marlin's moe_align_block_size format: the slots routed to this GPU's experts, grouped
    by expert (ascending slot within each) into BM-row blocks. sorted[b * BM + r] = slot (token * 8 + k), or
    n_slots for padding; block_expert[b] = local expert, or -1 unused; npad = rows used. Slots routed to
    other GPUs are left out (SGLang's groups them in a skipped block; nothing reads their rows). Also writes
    each slot's local expert (or -1): what SiLU-mul and the finalize key on. Replaces the expert-map index,
    its dtype copy and moe_align_block_size."""
    s = tl.arange(0, NS)
    s_ok = s < n_slots
    e = tl.load(ids_ptr + s, mask=s_ok, other=-1) - local_start
    local = s_ok & (e >= 0) & (e < COUNT)
    tl.store(local_ptr + s, tl.where(local, e, -1).to(tl.int32), mask=s_ok)
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
    rows = tl.arange(0, ROWS)  # ROWS: a power of two >= max_blocks * BM (BM can be 48)
    tl.store(sorted_ptr + rows, tl.zeros([ROWS], tl.int32) + n_slots, mask=rows < max_blocks * BM)
    tl.debug_barrier()
    tl.store(sorted_ptr + dest, s_sorted.to(tl.int32), mask=keep)
    b = tl.arange(0, NB)
    be = tl.sum(tl.where((b[:, None] >= block_start[None, :]) & (b[:, None] < block_end[None, :]),
                         ex[None, :], 0), 1)
    n_blocks = tl.max(block_end, 0)
    used = b < n_blocks
    tl.store(block_expert_ptr + b, tl.where(used, be, -1).to(tl.int32), mask=b < max_blocks)
    tl.store(npad_ptr, (n_blocks * BM).to(tl.int32))


def v2_prepare(weights):
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
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "post_norm_w": weights["post_norm_w"].contiguous(),
    }


def v2_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    # Rows per Marlin block: SGLang's heuristic, except 16 at 9-32 tokens (bs 16: 78 vs 92 us, probe_m3.py).
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    if 8 < num_tokens <= 32:
        block_size_m = 16
    topk_ids = data["topk_ids"].to(torch.int32).contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    n_slots = num_tokens * TOP_K
    max_blocks = triton.cdiv(n_slots, block_size_m) + bank["count"]
    sorted_ids = torch.empty((max_blocks * block_size_m,), dtype=torch.int32, device=x.device)
    block_experts = torch.empty((max_blocks,), dtype=torch.int32, device=x.device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=x.device)
    local_ids = torch.empty((num_tokens, TOP_K), dtype=torch.int32, device=x.device)
    _v2_align_kernel[(1,)](topk_ids, sorted_ids, block_experts, n_padded, local_ids, cfg["local_start"], n_slots, max_blocks,
                        COUNT=bank["count"], BINS=triton.next_power_of_2(bank["count"] + 1), BM=block_size_m,
                        NS=max(16, triton.next_power_of_2(n_slots)), NB=triton.next_power_of_2(max_blocks),
                        ROWS=triton.next_power_of_2(max_blocks * block_size_m), num_warps=4)

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
    # The post-expert norm and weighted sum in one kernel: SGLang's falls back to ~14 PyTorch kernels up to
    # 16 tokens. Slots routed off this GPU are skipped by expert id (their y rows are never written).
    out = torch.empty((num_tokens, size_k), device=x.device, dtype=torch.bfloat16)
    _v2_finalize_kernel[(num_tokens,)](y, local_ids, topk_weights, state["post_norm_w"], out,
                                    0, size_k, COUNT=bank["count"], TOPK=TOP_K,
                                    BH=triton.next_power_of_2(size_k), EPS=POST_NORM_EPS, num_warps=8)
    return out


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
    state = v2_prepare(weights)
    dev = weights["w13"].device
    W = {k: weights[k][:32] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    REP = {}
    for bn, bk in ((64, 128), (128, 128)):
        r13, r2 = repack(W["w13"], bk), repack(W["w2"], bk)
        REP[(bn, bk, False)] = (r13, W["w13_sf"], r2, W["w2_sf"])
        REP[(bn, bk, True)] = (tile_major(r13, bn, bk, 2), tile_major(W["w13_sf"], bn, bk, 16),
                               tile_major(r2, bn, bk, 2), tile_major(W["w2_sf"], bn, bk, 16))
    WS = make_workspace(8 * 256, 4096, 4096, dev)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = []

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
    for bs in (1, 8, 16, 32, 128, 256):
        if time.time() - t0 > 250:
            break
        data = _inputs(bs, 7, dev)
        ref = v2_fn(data, state).float()
        v2t = bench(lambda: v2_fn(data, state))
        res = []
        for tiled in (True, False):
            for bn in (64, 128):
                for warps in (4, 8):
                    for s1, s2 in ((1, 1), (2, 2), (4, 4), (8, 4)):
                        bm = 16 if bs <= 32 else 64
                        if time.time() - t0 > 250 or (bs > 32 and s1 > 2) or (bm * bn > 8192):
                            continue
                        w13, s13, w2, s2_ = REP[(bn, 128, tiled)]
                        tag = f"{'T' if tiled else 'R'}bn{bn},w{warps},s{s1}/{s2}"
                        fn = lambda w13=w13, s13=s13, w2=w2, s2_=s2_, bn=bn, warps=warps, s1=s1, s2=s2, bm=bm, tiled=tiled: moe_forward(
                            data["hidden_states"], data["topk_ids"], data["topk_weights"], w13, s13, W["w13_gs"], w2, s2_,
                            W["w2_gs"], weights["post_norm_w"], 0, 32, bn1=bn, bn2=bn, bk=128, warps=warps, stages=3, bm=bm,
                            fast=True, split1=s1, split2=s2, ws=WS, tiled=tiled)
                        try:
                            out = fn().float()
                            err = float((out - ref).norm() / ref.norm())
                            if not math.isfinite(err) or err > 0.02:
                                res.append((1e9, tag + f"BAD{err:.1e}"))
                                continue
                            t = bench(fn)
                            if float((fn().float() - ref).norm() / ref.norm()) > 0.02:
                                res.append((1e9, tag + "BAD-after-replays"))
                                continue
                            res.append((t, tag))
                        except Exception as e:  # noqa: BLE001
                            res.append((1e9, tag + type(e).__name__[:12] + str(e)[:60]))
        best_t = min((v for v, t in res if t.startswith("T")), default=float("nan"))
        best_r = min((v for v, t in res if t.startswith("R")), default=float("nan"))
        lines.append(f"bs{bs} v2 {v2t:.1f} best-tiled {best_t:.1f} best-rowmajor {best_r:.1f}")
        res.sort()
        bad = [t for v, t in res if v >= 1e9]
        lines.append(f"bs{bs} v2 {v2t:.1f} | triton " + " ".join(f"{v:.1f}/{t}" for v, t in res[:4])
                     + f" | bad {len(bad)} {bad[0] if bad else ''} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
