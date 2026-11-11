# okc: problem=muse_moe_w4a16
"""Probe D2, not a submission: CUDA mma.sync W4A16 grouped GEMM (lane-packed NVFP4 weights straight to registers, PTX
dequant, split-K) inside the MoE vs the v2 submission path (Marlin). Was probe M10:
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




def _codes(packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] uint8 (element 2i low nibble, 2i+1 high) -> [..., K] uint8 codes."""
    return torch.stack((packed & 15, packed >> 4), -1).flatten(-2)


def lane_pack(packed: torch.Tensor, sf: torch.Tensor):
    """packed [E, N, K/2] uint8, sf [E, N, K/16] float8_e4m3fn -> (codes [E, N/16, K/16, 32] int32 (uint32 bits),
    scales [E, N/16, K/16, 32] int16 (uint16 bits))."""
    e, n, kh = packed.shape
    k = kh * 2
    c = _codes(packed).view(e, n // 16, 16, k // 16, 16)            # [E, S, row, J, col]
    c = c.permute(0, 1, 3, 2, 4)                                     # [E, S, J, row(16), col(16)]
    lane = torch.arange(32, device=packed.device)
    g, t = lane // 4, lane % 4
    rows = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], -1)                        # a0..a7
    cols = torch.stack([2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9], -1)
    a = c[:, :, :, rows, cols].to(torch.int32)                       # [E, S, J, 32, 8]
    word = ((a[..., 0] | (a[..., 4] << 4))
            | ((a[..., 1] | (a[..., 5] << 4)) << 8)
            | ((a[..., 2] | (a[..., 6] << 4)) << 16)
            | ((a[..., 3] | (a[..., 7] << 4)) << 24))
    s = sf.view(torch.uint8).view(e, n // 16, 16, k // 16).permute(0, 1, 3, 2)  # [E, S, J, row]
    sg = s[..., g].to(torch.int32)                                    # [E, S, J, 32]
    sg8 = s[..., g + 8].to(torch.int32)
    scale = sg | (sg8 << 8)
    # int32 / int16 hold the uint32 / uint16 bit patterns (values above the signed range wrap; the bits are what count).
    return word.contiguous(), scale.to(torch.int16).contiguous()


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



CUDA_SRC = r"""// Grouped W4A16 (NVFP4 weights x BF16 activations) GEMM for the routed experts, mma.m16n8k16 with the weights as the
// A operand ("swap AB": weight rows are the MMA's m, tokens its n).
//
// Weights arrive lane-packed (w4a16_pack.lane_pack, lossless): per (expert, 16-row stripe, 16-wide k step) a warp reads
// 32 x uint32 of E2M1 codes (one lane's 8 A-fragment values) and 32 x uint16 of E4M3 scales (its rows g and g + 8;
// a k step is exactly one NVFP4 scale group). They go straight to registers, no shared memory. Dequant (exact): prmt
// two bytes into the register halves; mask/shift each code's sign to bit 15 and magnitude to bits 8..6 (value x 2^-126
// as BF16, 0.5 a subnormal); fma.rn.bf16x2 by 2^126, then by the row's scale. The per-expert FP32 global scale is
// applied to the accumulator.
//
// A CTA takes one expert block of BT = 8 NT routed slots (tokens), WARPS stripes (one per warp) and one 1/SPLIT share
// of K. The block's activations are staged chunk by chunk in padded shared memory (the B operand). With SPLIT > 1,
// partials meet in zeroed FP32 workspaces: the last CTA of a tile (arrival counter) applies the epilogue and re-zeroes.
// GATED: each warp also takes the matching "up" stripe (rows I + ...) and the epilogue is silu(g) * u.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace w4a16 {

constexpr int WARPS = 8;
constexpr int U = 8;                    // k steps of weight loads issued together per warp
// k steps per staged activation chunk, so that 8 NT rows x (16 KC + 8) bf16 stay ~17 KB of static shared memory.
template <int NT> struct Chunk { static constexpr int KC = NT <= 2 ? 32 : (NT == 4 ? 16 : 8); };

__device__ __forceinline__ uint32_t ld_stream_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint16_t ld_stream_u16(const uint16_t* p) {
  uint16_t v;
  asm volatile("ld.global.nc.L1::no_allocate.b16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}

// One E4M3 byte -> bf16x2 (s, s), exactly (E4M3 is a subset of BF16).
__device__ __forceinline__ uint32_t scale2(uint32_t byte) {
  __nv_fp8_e4m3 f;
  f.__x = static_cast<__nv_fp8_storage_t>(byte);
  const __nv_bfloat16 b = __float2bfloat16_rn(static_cast<float>(f));
  const uint32_t u = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&b));
  return u | (u << 16);
}

// The 4 A-fragment registers of one lane from its packed word and row scales (see w4a16_pack.py for the layout).
__device__ __forceinline__ void dequant(uint32_t w, uint32_t sg, uint32_t sg8, uint32_t (&a)[4]) {
  asm volatile(
      "{\n"
      ".reg .b32 t, x, y, z, bias, nz;\n"
      "mov.b32 z, 0;\n"
      "mov.b32 bias, 0x7E807E80;\n"
      "mov.b32 nz, 0x80008000;\n"
      // low nibbles of bytes 0,1 -> a0 (row g)
      "prmt.b32 t, %4, z, 0x4140;\n"
      "and.b32 x, t, 0x00070007;\n shl.b32 x, x, 6;\n and.b32 y, t, 0x00080008;\n shl.b32 y, y, 12;\n or.b32 x, x, y;\n"
      "fma.rn.bf16x2 x, x, bias, nz;\n fma.rn.bf16x2 %0, x, %5, nz;\n"
      // low nibbles of bytes 2,3 -> a1 (row g + 8)
      "prmt.b32 t, %4, z, 0x4342;\n"
      "and.b32 x, t, 0x00070007;\n shl.b32 x, x, 6;\n and.b32 y, t, 0x00080008;\n shl.b32 y, y, 12;\n or.b32 x, x, y;\n"
      "fma.rn.bf16x2 x, x, bias, nz;\n fma.rn.bf16x2 %1, x, %6, nz;\n"
      // high nibbles of bytes 0,1 -> a2 (row g)
      "prmt.b32 t, %4, z, 0x4140;\n"
      "and.b32 x, t, 0x00700070;\n shl.b32 x, x, 2;\n and.b32 y, t, 0x00800080;\n shl.b32 y, y, 8;\n or.b32 x, x, y;\n"
      "fma.rn.bf16x2 x, x, bias, nz;\n fma.rn.bf16x2 %2, x, %5, nz;\n"
      // high nibbles of bytes 2,3 -> a3 (row g + 8)
      "prmt.b32 t, %4, z, 0x4342;\n"
      "and.b32 x, t, 0x00700070;\n shl.b32 x, x, 2;\n and.b32 y, t, 0x00800080;\n shl.b32 y, y, 8;\n or.b32 x, x, y;\n"
      "fma.rn.bf16x2 x, x, bias, nz;\n fma.rn.bf16x2 %3, x, %6, nz;\n"
      "}\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(w), "r"(sg), "r"(sg8));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float silu_mul(float g, float u) { return g / (1.f + __expf(-g)) * u; }

template <int NT, bool GATED>
__global__ void __launch_bounds__(WARPS * 32) grouped_kernel(
    const uint32_t* __restrict__ wc, const uint16_t* __restrict__ wsc, const float* __restrict__ gscale,
    const __nv_bfloat16* __restrict__ x, int row_div, const int* __restrict__ sorted, const int* __restrict__ block_expert,
    __nv_bfloat16* __restrict__ out, float* __restrict__ ws0, float* __restrict__ ws1, int* __restrict__ cnt,
    int w_stripes, int out_n, int ksteps, int k, int split) {
  constexpr int BT = 8 * NT;
  constexpr int KC = Chunk<NT>::KC;
  constexpr int XS = KC * 16 + 8;  // padded bf16 row stride (conflict-free B-fragment reads)
  __shared__ __align__(16) __nv_bfloat16 xs[BT][XS];
  __shared__ int slots[BT];
  __shared__ int last;
  const int e = block_expert[blockIdx.x];
  if (e < 0) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rank = blockIdx.z;
  const int s0 = (int)((long long)rank * ksteps / split), s1 = (int)((long long)(rank + 1) * ksteps / split);
  const int stripe = blockIdx.y * WARPS + warp;               // output stripe (gate stripe if GATED)
  if (threadIdx.x < BT) slots[threadIdx.x] = sorted[blockIdx.x * BT + threadIdx.x];
  __syncthreads();  // slots visible to all (the epilogue reads them even if this rank has no k steps)

  const size_t base_g = ((size_t)e * w_stripes + stripe) * ksteps * 32 + lane;
  const size_t base_u = ((size_t)e * w_stripes + stripe + (GATED ? out_n / 16 : 0)) * ksteps * 32 + lane;

  float acc[NT][4], accu[GATED ? NT : 1][4];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acc[j][q] = 0.f;
      if (GATED) accu[GATED ? j : 0][q] = 0.f;
    }

  for (int cs = s0; cs < s1; cs += KC) {
    const int ce = min(cs + KC, s1);
    __syncthreads();  // the previous chunk's reads of xs are done
    // Stage the block's activations for k steps [cs, ce): rows past the block's slots as zero.
    const int pieces = BT * (ce - cs) * 2;  // 16-byte pieces (8 bf16)
    for (int p = threadIdx.x; p < pieces; p += WARPS * 32) {
      const int r = p / ((ce - cs) * 2), c8 = p % ((ce - cs) * 2);
      const int slot = slots[r];
      uint4 v = make_uint4(0, 0, 0, 0);
      if (slot >= 0) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)(slot / row_div) * k + (size_t)cs * 16) + c8);
      *reinterpret_cast<uint4*>(&xs[r][c8 * 8]) = v;
    }
    __syncthreads();
    for (int st0 = cs; st0 < ce; st0 += U) {
      uint32_t wg[U], sgw[U], wu[GATED ? U : 1], suw[GATED ? U : 1];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (st0 + u < ce) {
          wg[u] = ld_stream_u32(wc + base_g + (size_t)(st0 + u) * 32);
          sgw[u] = ld_stream_u16(wsc + base_g + (size_t)(st0 + u) * 32);
          if (GATED) {
            wu[GATED ? u : 0] = ld_stream_u32(wc + base_u + (size_t)(st0 + u) * 32);
            suw[GATED ? u : 0] = ld_stream_u16(wsc + base_u + (size_t)(st0 + u) * 32);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (st0 + u < ce) {
          uint32_t a[4];
          dequant(wg[u], scale2(sgw[u] & 0xFF), scale2(sgw[u] >> 8), a);
          uint32_t au[4];
          if (GATED) dequant(wu[GATED ? u : 0], scale2(suw[GATED ? u : 0] & 0xFF), scale2(suw[GATED ? u : 0] >> 8), au);
          const int kc = (st0 + u - cs) * 16 + 2 * t;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[j * 8 + g][kc]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[j * 8 + g][kc + 8]);
            mma(acc[j], a, b0, b1);
            if (GATED) mma(accu[GATED ? j : 0], au, b0, b1);
          }
        }
      }
    }
  }

  // C fragment: acc[j][0..1] = (weight row g, tokens 8j + 2t, +1), [2..3] = (row g + 8, same tokens).
  const float gsc = gscale[e];
  const int n0 = stripe * 16;
  auto finish = [&](float vg, float vu) -> float { return GATED ? silu_mul(vg * gsc, vu * gsc) : vg * gsc; };
  if (split == 1) {
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int slot = slots[j * 8 + 2 * t + (q & 1)];
        if (slot >= 0)
          out[(size_t)slot * out_n + n0 + g + ((q & 2) ? 8 : 0)] =
              __float2bfloat16_rn(finish(acc[j][q], GATED ? accu[GATED ? j : 0][q] : 0.f));
      }
    return;
  }
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int slot = slots[j * 8 + 2 * t + (q & 1)];
      if (slot >= 0) {
        const size_t o = (size_t)slot * out_n + n0 + g + ((q & 2) ? 8 : 0);
        atomicAdd(ws0 + o, acc[j][q]);
        if (GATED) atomicAdd(ws1 + o, accu[GATED ? j : 0][q]);
      }
    }
  __syncthreads();
  const int tile = blockIdx.x * gridDim.y + blockIdx.y;
  if (threadIdx.x == 0) {
    __threadfence();
    last = (atomicAdd(cnt + tile, 1) == split - 1);
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int slot = slots[j * 8 + 2 * t + (q & 1)];
      if (slot >= 0) {
        const size_t o = (size_t)slot * out_n + n0 + g + ((q & 2) ? 8 : 0);
        const float vg = __ldcg(ws0 + o);
        const float vu = GATED ? __ldcg(ws1 + o) : 0.f;
        out[o] = __float2bfloat16_rn(finish(vg, vu));
        ws0[o] = 0.f;  // this thread read it; its own position
        if (GATED) ws1[o] = 0.f;
      }
    }
  __syncthreads();
  if (threadIdx.x == 0) cnt[tile] = 0;
}

template <int NT, bool GATED>
static cudaError_t launch_t(const void* wc, const void* wsc, const float* gs, const void* x, int row_div, const int* sorted,
                            const int* block_expert, void* out, float* ws0, float* ws1, int* cnt, int nblocks, int w_stripes,
                            int out_n, int k, int split, cudaStream_t stream) {
  dim3 grid(nblocks, out_n / (16 * WARPS), split);
  grouped_kernel<NT, GATED><<<grid, WARPS * 32, 0, stream>>>(
      static_cast<const uint32_t*>(wc), static_cast<const uint16_t*>(wsc), gs, static_cast<const __nv_bfloat16*>(x),
      row_div, sorted, block_expert, static_cast<__nv_bfloat16*>(out), ws0, ws1, cnt, w_stripes, out_n, k / 16, k, split);
  return cudaGetLastError();
}

}  // namespace w4a16

// NT: 1, 2, 4 or 8 (block of 8 NT slots). Returns a cudaError_t code (0 = ok).
int w4a16_launch(const void* wc, const void* wsc, const float* gs, const void* x, int row_div, const int* sorted,
                 const int* block_expert, void* out, float* ws0, float* ws1, int* cnt, int nblocks, int w_stripes, int out_n,
                 int k, int split, int nt, bool gated, cudaStream_t stream) {
  using namespace w4a16;
#define W4A16_CASE(NTV)                                                                                              \
  if (nt == NTV)                                                                                                     \
    return gated ? (int)launch_t<NTV, true>(wc, wsc, gs, x, row_div, sorted, block_expert, out, ws0, ws1, cnt, nblocks, \
                                            w_stripes, out_n, k, split, stream)                                      \
                 : (int)launch_t<NTV, false>(wc, wsc, gs, x, row_div, sorted, block_expert, out, ws0, ws1, cnt,       \
                                             nblocks, w_stripes, out_n, k, split, stream);
  W4A16_CASE(1)
  W4A16_CASE(2)
  W4A16_CASE(4)
  W4A16_CASE(8)
#undef W4A16_CASE
  return (int)cudaErrorInvalidValue;
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
int w4a16_launch(const void* wc, const void* wsc, const float* gs, const void* x, int row_div, const int* sorted,
                 const int* block_expert, void* out, float* ws0, float* ws1, int* cnt, int nblocks, int w_stripes, int out_n,
                 int k, int split, int nt, bool gated, cudaStream_t stream);
void w4a16_run(torch::Tensor wc, torch::Tensor wsc, torch::Tensor gs, torch::Tensor x, int64_t row_div, torch::Tensor sorted,
               torch::Tensor block_expert, torch::Tensor out, torch::Tensor ws0, torch::Tensor ws1, torch::Tensor cnt,
               int64_t nblocks, int64_t w_stripes, int64_t out_n, int64_t k, int64_t split, int64_t nt, bool gated) {
  int err = w4a16_launch(wc.data_ptr(), wsc.data_ptr(), gs.data_ptr<float>(), x.data_ptr(), (int)row_div,
                         sorted.data_ptr<int>(), block_expert.data_ptr<int>(), out.data_ptr(), ws0.data_ptr<float>(),
                         ws1.data_ptr<float>(), cnt.data_ptr<int>(), (int)nblocks, (int)w_stripes, (int)out_n, (int)k,
                         (int)split, (int)nt, gated, c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == 0, "w4a16_launch: ", cudaGetErrorString((cudaError_t)err));
}
"""
CPP_SRC = ("void w4a16_run(torch::Tensor wc, torch::Tensor wsc, torch::Tensor gs, torch::Tensor x, int64_t row_div, "
           "torch::Tensor sorted, torch::Tensor block_expert, torch::Tensor out, torch::Tensor ws0, torch::Tensor ws1, "
           "torch::Tensor cnt, int64_t nblocks, int64_t w_stripes, int64_t out_n, int64_t k, int64_t split, int64_t nt, "
           "bool gated);")


def prepare(weights):
    t0 = time.time()
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="okc_w4a16_d2", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["w4a16_run"],
                      extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    lines = [f"compile {time.time() - t0:.0f}s"]
    state = v2_prepare(weights)
    dev = weights["w13"].device
    W = {k: weights[k][:32] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    c13, s13 = lane_pack(W["w13"], W["w13_sf"])
    c2, s2 = lane_pack(W["w2"], W["w2_sf"])
    gs13 = W["w13_gs"].float().contiguous()
    gs2 = W["w2_gs"].float().contiguous()
    H = I = 4096
    max_slots = 256 * 8
    ws_g = torch.zeros(max_slots * I, device=dev); ws_u = torch.zeros(max_slots * I, device=dev)
    ws_y = torch.zeros(max_slots * H, device=dev); cnt1 = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
    cnt2 = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
    lines.append(f"pack + prep [{time.time() - t0:.0f}s]")
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)

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

    def run(data, nt, sp1, sp2):
        x, ids, wts = data["hidden_states"], data["topk_ids"], data["topk_weights"]
        m = x.shape[0]
        n_slots = m * 8
        bt = 8 * nt
        nblocks = triton.cdiv(n_slots, bt) + 32
        sorted_ids = torch.empty((nblocks * bt,), dtype=torch.int32, device=dev)
        block_expert = torch.empty((nblocks,), dtype=torch.int32, device=dev)
        _align_kernel[(1,)](ids, sorted_ids, block_expert, 0, n_slots, nblocks, COUNT=32, BINS=64, BM=bt,
                            NS=max(16, triton.next_power_of_2(n_slots)), NB=triton.next_power_of_2(nblocks), num_warps=4)
        inter = torch.empty((n_slots, I), dtype=torch.bfloat16, device=dev)
        mod.w4a16_run(c13, s13, gs13, x, 8, sorted_ids, block_expert, inter, ws_g, ws_u, cnt1, nblocks, 2 * I // 16, I, H,
                      sp1, nt, True)
        y = torch.empty((n_slots, H), dtype=torch.bfloat16, device=dev)
        mod.w4a16_run(c2, s2, gs2, inter, 1, sorted_ids, block_expert, y, ws_y, ws_y, cnt2, nblocks, H // 16, H, I,
                      sp2, nt, False)
        out = torch.empty((m, H), dtype=torch.bfloat16, device=dev)
        _finalize_kernel[(m,)](y, ids, wts, weights["post_norm_w"], out, 0, H, COUNT=32, TOPK=8,
                               BH=triton.next_power_of_2(H), EPS=1e-8, num_warps=8)
        return out

    nop = torch.zeros(1, device=dev)
    lines.append(f"nop {bench(lambda: nop.add_(1)):.1f}")
    for bs in (1, 8, 16, 32, 128, 256):
        if time.time() - t0 > 270:
            lines.append("TIME BUDGET HIT")
            break
        data = _inputs(bs, 7, dev)
        ref = v2_fn(data, state).float()
        v2t = bench(lambda: v2_fn(data, state))
        res = []
        nts = (1, 2) if bs <= 8 else (2, 4) if bs <= 32 else (4, 8)
        for nt in nts:
            for sp1, sp2 in ((1, 1), (2, 2), (4, 4), (8, 4), (8, 8)):
                if bs >= 128 and sp1 > 2:
                    continue
                tag = f"nt{nt},s{sp1}/{sp2}"
                try:
                    fn = lambda nt=nt, sp1=sp1, sp2=sp2: run(data, nt, sp1, sp2)
                    out = fn().float()
                    err = float((out - ref).norm() / ref.norm())
                    if not math.isfinite(err) or err > 0.02:
                        res.append((1e9, tag + f"BAD{err:.1e}"))
                        continue
                    t = bench(fn)
                    err2 = float((fn().float() - ref).norm() / ref.norm())
                    if err2 > 0.02:
                        res.append((1e9, tag + f"BAD-after-replays{err2:.1e}"))
                        continue
                    res.append((t, tag + f",err{err:.1e}"))
                except Exception as e:  # noqa: BLE001
                    res.append((1e9, tag + type(e).__name__[:12] + str(e)[:100]))
        res.sort()
        bad = [t for v, t in res if v >= 1e9]
        lines.append(f"bs{bs} v2 {v2t:.1f} | cuda " + " ".join(f"{v:.1f}/{t}" for v, t in res[:4])
                     + f" | bad {len(bad)} {bad[0] if bad else ''} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
