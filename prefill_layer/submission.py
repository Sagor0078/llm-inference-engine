# okc: problem=muse_prefill_layer
"""One Muse Spark Open decoder layer for prefill (TP8/EP8 on one H200): the muse_decode_layer v1 file, with
SGLang's FA3 attention path restored for blocks longer than 8 tokens (all prefill blocks: muse_attention_decode's
kernel holds every query row of a KV head in one tile).

  qkv_gate, o, pooler, unpooler    muse_bf16_proj's split-K swap-AB TMA Triton kernels (m <= 32, and
                                   the pooler at 65-128 tokens; cuBLAS elsewhere, as before)
  QK-norm .. gated output norm     muse_attention_decode's three Triton kernels: prologue (norm, RoPE,
                                   FP8 cache write, E4M3 query), split-KV paged attention, combine +
                                   RMSNorm x sigmoid(gate)

Everything else is SGLang's, unchanged: the router (its FP32 input up to 8 tokens), routing, the
Marlin NVFP4 experts and the norm/residual boundaries."""



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
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
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
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


# ---- muse_bf16_proj v5 (submission_v5_good.py) ----

WS_FLOATS = 1 << 20
WS_TILES = 4096
SKINNY_MAX_M = 32

# (BN, BK, num_warps, num_stages, SPLIT_K) per BF16 layer for m <= 32, swap-AB TMA kernel.
BF16_CONFIGS = {
    "qkv_gate": (64, 128, 4, 4, 7),
    "o": (64, 256, 4, 4, 1),
    "pooler": (64, 128, 4, 3, 4),
    "unpooler_slice": (64, 128, 4, 3, 16),  # m <= 8: this GPU's 1024-row slice
    "unpooler": (64, 256, 4, 4, 1),
    "pooler_m128": (64, 128, 4, 4, 2),  # 64 < m <= 128: the one wider shape that beats cuBLAS
}


@triton.jit
def _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, tile, SPLIT_K):
    """Write a unit's partial tile: directly, or into the workspace with the last unit reducing."""
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
        # All of this unit's adds land before it counts itself in; the last one in reduces.
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            # The others are done with this tile; read past L1 (the adds landed in L2), then re-zero.
            total = tl.load(ws_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=mask)
            # Every thread has read before any re-zeroes: the zeros can land in another thread's
            # layout than the loads (BM 128 on 8 warps lost partial sums without this).
            tl.debug_barrier()
            tl.store(ws_ptrs, tl.zeros_like(acc), mask=mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _swab_kernel(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[:, n-tile]^T = W[n-tile] @ x^T over this unit's k-slice; W and x through TMA."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    # This unit's share of the K chunks, balanced for any SPLIT_K.
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for c in range(c0, c1):
        x = x_desc.load([pid_m * BM, c * BK])  # rows past m read as zero
        w = w_desc.load([pid_n * BN, c * BK])
        acc = tl.dot(w, tl.trans(x), acc)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    ws_ptrs = ws_ptr + offs_m[None, :] * N + offs_n[:, None]
    _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K)


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _router_kernel(x_ptr, hi_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_xm, stride_wn,
                   stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out = x @ (hi + lo)^T in FP32, both BF16 halves through the tensor cores."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = c0 * BK + tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(c0, c1):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        hi = tl.load(hi_ptr + w_off, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(hi), acc)
        lo = tl.load(lo_ptr + w_off, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(lo), acc)
        x_ptrs += BK
        w_off += BK
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
    _finish(acc, out_ptrs, ws_ptrs, m_mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K)


def _router_config(m: int, n: int):
    """(BM, BN, BK, num_warps, num_stages, SPLIT_K) for the router at m tokens. Shape-only."""
    if m <= 16:
        return (16, 32, 128, 4, 4, 16)
    if m <= 32:
        return (32, 64, 128, 4, 4, 33)
    if m <= 128:
        return (64, 32, 128, 4, 3, 16)
    tiles = triton.cdiv(m, 64) * (n // 64)
    return (64, 64, 128, 4, 4, max(1, min(8, 264 // tiles)))


def _workspace(state, m: int, n: int, tiles: int, split: int, device):
    if split == 1 or (m * n <= WS_FLOATS and tiles <= WS_TILES):
        return state["_ws"], state["_cnt"]
    # Outside the persistent workspace's reach: a fresh zeroed one (shape-based, graph-safe).
    return (torch.zeros(m * n, device=device, dtype=torch.float32),
            torch.zeros(tiles, device=device, dtype=torch.int32))


def _router(x: torch.Tensor, n: int, state) -> torch.Tensor:
    m, k = x.shape
    bm, bn, bk, warps, stages, split = _router_config(m, n)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    tiles = triton.cdiv(m, bm) * (n // bn)
    ws, cnt = _workspace(state, m, n, tiles, split, x.device)
    hi, lo = state["_router_hi"], state["_router_lo"]
    _router_kernel[(n // bn, split, triton.cdiv(m, bm))](
        x, hi, lo, out, ws, cnt, m, n, k, split, x.stride(0), hi.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def _swab(x: torch.Tensor, layer: str, n: int, state) -> torch.Tensor:
    m, k = x.shape
    key = "unpooler_slice" if layer == "unpooler" and n <= 1024 else layer
    if m > SKINNY_MAX_M:
        key = "pooler_m128"
    bn, bk, warps, stages, split = BF16_CONFIGS[key]
    bm = max(16, triton.next_power_of_2(m))
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    tiles = n // bn
    ws, cnt = _workspace(state, m, n, tiles, split, x.device)
    w_desc = state["_desc"][(layer, n)]
    x_desc = TensorDescriptor.from_tensor(x, [bm, bk])
    _swab_kernel[(n // bn, split, 1)](w_desc, x_desc, out, ws, cnt, m, n, k, split, out.stride(0),
                                       BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def proj_prepare(weights):
    """Weights as given, their TMA descriptors, the router's BF16 hi + lo split, a zeroed split-K
    workspace, and every kernel compiled ahead of the timed calls."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["_router_hi"] = hi
    state["_router_lo"] = (router - hi.float()).to(torch.bfloat16)
    device = router.device
    state["_ws"] = torch.zeros(WS_FLOATS, device=device, dtype=torch.float32)
    state["_cnt"] = torch.zeros(WS_TILES, device=device, dtype=torch.int32)
    state["_desc"] = {}
    for key, (bn, bk, *_) in BF16_CONFIGS.items():
        layer = key.split("_")[0] if key != "qkv_gate" else key
        n = 1024 if key == "unpooler_slice" else state[layer].shape[0]
        state["_desc"][(layer, n)] = TensorDescriptor.from_tensor(state[layer][:n], [bn, bk])
    for layer, w in weights.items():
        n_full, k = w.shape
        for m in (1, 5, 8, 9, 16, 32, 33, 64, 128, 256, 512, 1000):
            n = 1024 if layer == "unpooler" and m <= 8 else n_full
            x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
            proj_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return state


def proj_fn(data, state) -> torch.Tensor:
    """x @ W[:n].T: bfloat16, or float32 for the router."""
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    if x.stride(1) != 1 or x.data_ptr() % 16:
        x = x.contiguous().clone()
    m, k = x.shape
    if layer == "router":
        if n % 64 or k % 256:
            return x.float() @ state["router"][:n].T
        return _router(x, n, state)
    swab = m <= SKINNY_MAX_M or (layer == "pooler" and 64 < m <= 128)
    if swab and (layer, n) in state["_desc"] and k % 256 == 0:
        return _swab(x, layer, n, state)
    return x @ state[layer][:n].T


# ---- muse_attention_decode v7 (submission_v7_good.py) ----

RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
# Aim for about this many attention programs (8 per SM), from the shapes alone.
TARGET_PROGRAMS = 1056
# At most this many chunks per request: past it the combine's cost outgrows the parallelism.
MAX_SPLITS = 256
_INV_FREQ = {}


def _inv_freq(theta: float, d: int, device) -> torch.Tensor:
    """theta^(-2i/d), computed exactly as the reference does (FP32)."""
    key = (theta, d, str(device))
    if key not in _INV_FREQ:
        _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
    return _INV_FREQ[key]


@triton.jit
def _prologue_kernel(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
                     q8_ptr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
                     EPS: tl.constexpr):
    tok = tl.program_id(0)
    row = qkv_ptr + tok * ((2 * HQ + 2 * HKV) * D)
    half = tl.arange(0, D // 2)
    pos = tl.load(pos_ptr + tok).to(tl.float32)
    if ROPE:
        ang = pos * tl.load(inv_freq_ptr + half)  # FP32, as the reference: pos.float() * inv_freq
        cos = tl.cos(ang)[None, :]
        sin = tl.sin(ang)[None, :]

    # Queries: [HQ, D/2] even and odd halves of each head.
    hq = tl.arange(0, HQ)[:, None]
    qe = tl.load(row + hq * D + 2 * half[None, :]).to(tl.float32)
    qo = tl.load(row + hq * D + 2 * half[None, :] + 1).to(tl.float32)
    inv = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D + EPS)[:, None]
    qe, qo = qe * inv, qo * inv
    if ROPE:
        qe, qo = qe * cos - qo * sin, qe * sin + qo * cos
    # E4M3 rounds the normalized query before the 5.473 factor (folded into the softmax scale).
    qdst = q8_ptr + tok * (HQ * D) + hq * D + 2 * half[None, :]
    tl.store(qdst, qe.to(tl.float8e4nv))
    tl.store(qdst + 1, qo.to(tl.float8e4nv))

    # New keys: normalized (and rotated) like the queries; values as they come.
    hk = tl.arange(0, HKV)[:, None]
    kbase = row + HQ * D
    ke = tl.load(kbase + hk * D + 2 * half[None, :]).to(tl.float32)
    ko = tl.load(kbase + hk * D + 2 * half[None, :] + 1).to(tl.float32)
    kinv = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / D + EPS)[:, None]
    ke, ko = ke * kinv, ko * kinv
    if ROPE:
        ke, ko = ke * cos - ko * sin, ke * sin + ko * cos
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    loc = tl.load(loc_ptr + tok)
    kdst = k_cache_ptr + loc * (HKV * D) + hk * D + 2 * half[None, :]
    tl.store(kdst, (ke / k_scale).to(tl.float8e4nv))
    tl.store(kdst + 1, (ko / k_scale).to(tl.float8e4nv))
    d = tl.arange(0, D)[None, :]
    v = tl.load(row + (HQ + HKV) * D + hk * D + d).to(tl.float32)
    tl.store(v_cache_ptr + loc * (HKV * D) + hk * D + d, (v / v_scale).to(tl.float8e4nv))


@triton.jit
def _attn_kernel(q8_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, page_table_ptr, seqlens_ptr,
                 m_ptr, l_ptr, acc_ptr, max_pages, num_splits, chunk, qk_scale,
                 L: tl.constexpr, G: tl.constexpr, HP: tl.constexpr, M: tl.constexpr, HKV: tl.constexpr,
                 D: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr, BLOCK_N: tl.constexpr,
                 PV_FP8: tl.constexpr, PREFETCH: tl.constexpr):
    """One (request, HP adjacent KV heads, key chunk). With HP = 2 a gathered cache row is the full
    128 bytes of both heads, and one block-diagonal query tile covers both: head hp's queries sit in
    columns hp * D .. hp * D + D - 1 (zeros elsewhere), so a single dot gives each row its own head's
    logits, and of P @ V each row keeps its own head's half (the combine reads that half)."""
    bh = tl.program_id(0)  # b * (HKV / HP) + head pair
    split = tl.program_id(1)
    b = bh // (HKV // HP)
    h0 = (bh % (HKV // HP)) * HP
    total = tl.load(seqlens_ptr + b)  # keys in the request, the block's own included
    seq = total - L                   # the block's first position
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(seq - WINDOW + 1, 0)
    start = lo + split * chunk
    end = tl.minimum(start + chunk, total)

    # Query rows: token j x head-in-pair hp x group head g, row = (j * HP + hp) * G + g.
    r = tl.arange(0, M)
    j = r // (G * HP)
    hp = (r // G) % HP
    g = r % G
    row_ok = r < L * G * HP
    c = tl.arange(0, HP * D)  # columns: head hp's D values at hp * D
    q = tl.load(q8_ptr + ((b * L + j) * (HKV * G) + (h0 + hp) * G + g)[:, None] * D + (c % D)[None, :],
                mask=row_ok[:, None] & ((c // D)[None, :] == hp[:, None]), other=0.0)
    qpos = seq + j  # each row's position

    m_i = tl.full([M], -float("inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, HP * D], tl.float32)
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    scale = qk_scale * k_scale * 1.4426950408889634
    # PREFETCH: the page indices run one block ahead of the K/V loads that need them, so a block's
    # gather addresses never wait on a load issued in the same iteration (decode gains; verify,
    # which is compute-bound, measured slower with it).
    t_next = start + tl.arange(0, BLOCK_N)
    page_next = tl.load(page_table_ptr + b * max_pages + t_next // PAGE, mask=t_next < end, other=0)
    for n0 in range(start, end, BLOCK_N):
        t = n0 + tl.arange(0, BLOCK_N)
        t_ok = t < end
        if PREFETCH:
            page = page_next
            t_next = t + BLOCK_N
            page_next = tl.load(page_table_ptr + b * max_pages + t_next // PAGE, mask=t_next < end, other=0)
        else:
            page = tl.load(page_table_ptr + b * max_pages + t // PAGE, mask=t_ok, other=0)
        krow = (page.to(tl.int64) * PAGE + t % PAGE) * (HKV * D) + h0 * D
        # Rows sit at multiples of D bytes; saying so lets the gather use 16-byte vector loads
        # (the offsets come from loaded page indices, which alignment analysis can't see through).
        krow = tl.multiple_of(krow, D)
        kt = tl.load(k_cache_ptr + krow[:, None] + c[None, :], mask=t_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kt)) * scale  # [M, BLOCK_N], log2 units
        vis = t_ok[None, :] & (t[None, :] <= qpos[:, None])
        if WINDOW > 0:
            vis = vis & (t[None, :] > qpos[:, None] - WINDOW)
        s = tl.where(vis, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vt = tl.load(v_cache_ptr + krow[:, None] + c[None, :], mask=t_ok[:, None], other=0.0)
        if PV_FP8:  # FP8 P @ V on the tensor cores, as FA3 does with an FP8 cache
            pv = tl.dot(p.to(tl.float8e4nv), vt)
        else:
            pv = tl.dot(p.to(tl.bfloat16), vt.to(tl.bfloat16))
        acc = acc * alpha[:, None] + pv
        m_i = m_new
    # Partials, [B * HKV / HP, num_splits, M] (+ HP * D for acc); empty chunks leave m = -inf, l = 0.
    base = bh * num_splits + split
    tl.store(m_ptr + base * M + r, m_i)
    tl.store(l_ptr + base * M + r, l_i)
    tl.store(acc_ptr + (base * M + r)[:, None] * (HP * D) + c[None, :], acc * v_scale)


@triton.jit
def _combine_kernel(m_ptr, l_ptr, acc_ptr, qkv_ptr, out_ptr, num_splits,
                    L: tl.constexpr, G: tl.constexpr, HP: tl.constexpr, M: tl.constexpr, HQ: tl.constexpr,
                    HKV: tl.constexpr, D: tl.constexpr, SB: tl.constexpr, EPS: tl.constexpr):
    """One (token, query head): the chunks merged SB at a time (running max), then the gated norm."""
    tok = tl.program_id(0)  # b * L + j
    hq = tl.program_id(1)
    b = tok // L
    j = tok % L
    h = hq // G
    rr = (j * HP + h % HP) * G + hq % G
    d = tl.arange(0, D)
    mx = -float("inf")
    den = 0.0
    num = tl.zeros([D], tl.float32)
    for s0 in range(0, num_splits, SB):
        s = s0 + tl.arange(0, SB)
        s_ok = s < num_splits
        base = (b * (HKV // HP) + h // HP) * num_splits + s
        m = tl.load(m_ptr + base * M + rr, mask=s_ok, other=-float("inf"))
        l = tl.load(l_ptr + base * M + rr, mask=s_ok, other=0.0)
        acc = tl.load(acc_ptr + (base * M + rr)[:, None] * (HP * D) + (h % HP) * D + d[None, :], mask=s_ok[:, None],
                      other=0.0)
        mx_new = tl.maximum(mx, tl.max(m, 0))
        safe = tl.where(mx_new == -float("inf"), 0.0, mx_new)
        w = tl.where(m == -float("inf"), 0.0, tl.exp2(m - safe))
        r = tl.where(mx == -float("inf"), 0.0, tl.exp2(mx - safe))
        den = den * r + tl.sum(l * w, 0)
        num = num * r + tl.sum(acc * w[:, None], 0)
        mx = mx_new
    o = num / den
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + EPS)
    gate = tl.load(qkv_ptr + tok * ((2 * HQ + 2 * HKV) * D) + (HQ + 2 * HKV) * D + hq * D + d).to(tl.float32)
    o = o * tl.sigmoid(gate)
    tl.store(out_ptr + tok * (HQ * D) + hq * D + d, o.to(tl.bfloat16))


def _split_plan(units: int, capacity: int, block_n: int, target: int, max_splits: int):
    """(chunk, num_splits) from the shapes: about `target` programs at full capacity, at most
    `max_splits` chunks per request."""
    want = max(1, min(max_splits, target // units))
    chunk = max(block_n, triton.next_power_of_2(triton.cdiv(capacity, want)))
    return chunk, triton.cdiv(capacity, chunk)


def attn_fn(data) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    window = cfg["window"]
    group = hq // hkv
    tokens = batch * length
    qkv = data["qkv"]
    device = qkv.device

    rope = bool(cfg["rope_theta"])
    inv_freq = _inv_freq(cfg["rope_theta"] or 1.0, d, device)
    q8 = torch.empty((tokens, hq, d), device=device, dtype=torch.float8_e4m3fn)
    _prologue_kernel[(tokens,)](qkv, data["positions"], data["cache_loc"], inv_freq, data["k_cache"],
                                data["v_cache"], data["k_scale"], data["v_scale"], q8,
                                HQ=hq, HKV=hkv, D=d, ROPE=rope, EPS=RMS_EPS, num_warps=4)

    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    capacity = max_pages * page
    if window:
        capacity = min(capacity, window + length - 1)
    # Decode with a batch: both KV heads of a pair per program (whole 128-byte rows); verify's
    # 64-row tiles are compute-heavy, so there one head per program. Tile and pipeline depth per
    # layer and q_len from a sweep on the judge's H200 (probe_a1.py).
    hp = 2 if length == 1 and batch > 1 and hkv % 2 == 0 else 1
    block_n = 128 if (window == 0 and (length == 1 or batch == 1)) else 64
    stages = 3 if (window == 0 and length == 1) else 2
    units = batch * hkv // hp
    # Program target and split cap per layer and q_len from a whole-call sweep (probe_a3.py).
    target = TARGET_PROGRAMS // 2 if window and length == 1 else TARGET_PROGRAMS
    max_splits = 32 if length > 1 and batch > 1 else MAX_SPLITS
    chunk, num_splits = _split_plan(units, capacity, block_n, target, max_splits)
    rows = max(16, triton.next_power_of_2(length * group * hp))
    m = torch.empty((units, num_splits, rows), device=device, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty((units, num_splits, rows, hp * d), device=device, dtype=torch.float32)
    _attn_kernel[(units, num_splits)](
        q8, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["cache_seqlens"],
        m, l, acc, max_pages, num_splits, chunk, cfg["sm_scale"] * cfg["qk_scale_factor"],
        L=length, G=group, HP=hp, M=rows, HKV=hkv, D=d, PAGE=page, WINDOW=window, BLOCK_N=block_n,
        PV_FP8=False, PREFETCH=length == 1, num_warps=4, num_stages=stages)

    out = torch.empty((batch, length, hq * d), device=device, dtype=torch.bfloat16)
    _combine_kernel[(tokens, hq)](m, l, acc, qkv, out, num_splits, L=length, G=group, HP=hp, M=rows, HQ=hq, HKV=hkv, D=d,
                                  SB=32, EPS=RMS_EPS, num_warps=4)
    return out


# ---- the layer: SGLang's MuseSpark1xDecoderLayer with the stages above ----

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
# From this many tokens the experts run as an in-call exact NVFP4 -> BF16 dequant of the local experts plus grouped
# BF16 GEMMs (torch._grouped_mm): at prefill sizes that beats Marlin's in-GEMM dequant by ~2x on the GEMMs.
GROUPED_MIN_TOKENS = 8192
GROUP_ALIGN = 16  # rows per expert group are padded to this
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _proj(x, layer, n, state):
    """x @ W[:n].T through muse_bf16_proj's kernels."""
    return proj_fn({"x": x, "config": {"m": x.shape[0], "n": n, "k": x.shape[1], "layer": layer}}, state["proj"])


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
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it (the FA3 path's RoPE).
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
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
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "grouped": _grouped_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "dequant_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "proj": proj_prepare({name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler", "router")}),
    }
    torch.cuda.synchronize()
    return state


@triton.jit
def _dequant_kernel(p_ptr, sf_ptr, out_ptr, n_bytes, BLOCK: tl.constexpr):
    """Packed E2M1 bytes x E4M3 block scales -> BF16 (exact: the global scale is applied after the GEMM)."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n_bytes
    b = tl.load(p_ptr + i, mask=ok, other=0).to(tl.int32)
    s = tl.load(sf_ptr + i // 8, mask=ok, other=0.0).to(tl.float32)
    s = tl.reshape(tl.join(s, s), [2 * BLOCK])
    c = tl.reshape(tl.join(b & 15, b >> 4), [2 * BLOCK])  # element 2i is the low nibble
    mag = c & 7
    big = ((mag + 252) << 22).to(tl.float32, bitcast=True)  # 1, 1.5, 2, 3, 4, 6 for codes 2-7
    v = tl.where(mag >= 2, big, mag.to(tl.float32) * 0.5)
    v = tl.where((c & 8) != 0, -v, v) * s
    o = tl.program_id(0).to(tl.int64) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + o, v.to(tl.bfloat16), mask=o < 2 * n_bytes)


@triton.jit
def _count_kernel(ids_ptr, cnt_ptr, n_slots, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """cnt[e] += routed slots of local expert e."""
    s = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(ids_ptr + s, mask=s < n_slots, other=-1)
    e = tl.where((e >= 0) & (e < COUNT), e, COUNT)
    hist = tl.histogram(e, 2 * COUNT)
    ex = tl.arange(0, 2 * COUNT)
    tl.atomic_add(cnt_ptr + ex, hist, mask=(ex < COUNT) & (hist > 0))


@triton.jit
def _scatter_kernel(x_ptr, ids_ptr, cnt_ptr, fill_ptr, a_ptr, slot_row_ptr, offs_ptr,
                    D: tl.constexpr, TOPK: tl.constexpr, COUNT: tl.constexpr, ALIGN: tl.constexpr):
    """One token: copies x[t] to its row in each local expert's group (groups padded to ALIGN rows, in expert
    order) and records the row per slot. offs = each group's cumulative end, as torch._grouped_mm takes it.
    Rows within a group are in arrival order; every row is computed independently, so that doesn't matter."""
    t = tl.program_id(0)
    ex = tl.arange(0, COUNT)
    counts = tl.load(cnt_ptr + ex)
    padded = (counts + ALIGN - 1) // ALIGN * ALIGN
    end = tl.cumsum(padded, 0)
    start = end - padded
    if t == 0:
        tl.store(offs_ptr + ex, end)
    d = tl.arange(0, D)
    x = tl.load(x_ptr + t.to(tl.int64) * D + d)
    for k in tl.static_range(TOPK):
        e = tl.load(ids_ptr + t * TOPK + k)
        if (e >= 0) & (e < COUNT):
            row = tl.sum(tl.where(ex == e, start, 0), 0) + tl.atomic_add(fill_ptr + e, 1)
            tl.store(a_ptr + row.to(tl.int64) * D + d, x)
            tl.store(slot_row_ptr + t * TOPK + k, row)


@triton.jit
def _silu_mul_kernel(h_ptr, offs_ptr, gs_ptr, out_ptr, I: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """Row r of expert e's group: silu(g * gs[e]) * (u * gs[e]), each scaled half rounded to BF16 as the GEMM
    output it stands for. Rows past the last group are skipped."""
    r = tl.program_id(0)
    end = tl.load(offs_ptr + tl.arange(0, COUNT))
    if r < tl.max(end, 0):
        e = tl.sum((end <= r).to(tl.int32), 0)
        gs = tl.load(gs_ptr + e)
        c = tl.arange(0, BLOCK)
        base = r.to(tl.int64) * (2 * I)
        g = (tl.load(h_ptr + base + c).to(tl.float32) * gs).to(tl.bfloat16).to(tl.float32)
        u = (tl.load(h_ptr + base + I + c).to(tl.float32) * gs).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + r.to(tl.int64) * I + c, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16))


@triton.jit
def _grouped_finalize_kernel(y_ptr, ids_ptr, slot_row_ptr, wts_ptr, gs_ptr, norm_w_ptr, out_ptr,
                             H: tl.constexpr, TOPK: tl.constexpr, COUNT: tl.constexpr, EPS: tl.constexpr):
    """out[t] = sum over t's local slots of weight * rmsnorm(bf16(y * gs[e]) * (1 + w))."""
    t = tl.program_id(0)
    h = tl.arange(0, H)
    w1 = 1.0 + tl.load(norm_w_ptr + h).to(tl.float32)
    out = tl.zeros([H], tl.float32)
    for k in range(TOPK):
        e = tl.load(ids_ptr + t * TOPK + k)
        if (e >= 0) & (e < COUNT):
            row = tl.load(slot_row_ptr + t * TOPK + k).to(tl.int64)
            y = (tl.load(y_ptr + row * H + h).to(tl.float32) * tl.load(gs_ptr + e)).to(tl.bfloat16)
            z = y.to(tl.float32) * w1
            inv = tl.rsqrt(tl.sum(z * z, 0) / H + EPS)
            out += tl.load(wts_ptr + t * TOPK + k) * z * inv
    tl.store(out_ptr + t.to(tl.int64) * H + h, out.to(tl.bfloat16))


def _grouped_bank(weights):
    """The local experts' NVFP4 weights as given, their per-expert global scales, and BF16 scratch the call
    dequantizes them into (no weight is kept in BF16 between calls)."""
    e, n13, half = weights["w13"].shape
    k = half * 2
    device = weights["w13"].device
    return {
        "w13": weights["w13"].contiguous(), "w13_sf": weights["w13_sf"].contiguous(),
        "w2": weights["w2"].contiguous(), "w2_sf": weights["w2_sf"].contiguous(),
        "w13_gs": weights["w13_gs"].float().reshape(-1).expand(e).contiguous(),
        "w2_gs": weights["w2_gs"].float().reshape(-1).expand(e).contiguous(),
        "w13_bf16": torch.empty((e, n13, k), device=device, dtype=torch.bfloat16),
        "w2_bf16": torch.empty((e, weights["w2"].shape[1], weights["w2"].shape[2] * 2), device=device,
                               dtype=torch.bfloat16),
        "post_expert_raw": weights["post_expert_norm"].contiguous(),
    }


def _dequant(packed, sf, out):
    n = packed.numel()
    _dequant_kernel[(triton.cdiv(n, 4096),)](packed, sf, out, n, BLOCK=4096, num_warps=8)


def _dequant_experts(state):
    g = state["grouped"]
    _dequant(g["w13"], g["w13_sf"], g["w13_bf16"])
    _dequant(g["w2"], g["w2_sf"], g["w2_bf16"])


def _grouped_experts(x, topk_weights, local_ids, state):
    """The experts as grouped BF16 GEMMs on the call's dequantized weights, for prefill-size token counts."""
    g = state["grouped"]
    num_tokens, size_k = x.shape
    size_n = g["w2_bf16"].shape[2]
    device = x.device
    n_slots = num_tokens * TOP_K
    max_rows = n_slots + LOCAL_EXPERTS * (GROUP_ALIGN - 1)
    counters = torch.zeros((2, LOCAL_EXPERTS), device=device, dtype=torch.int32)
    offs = torch.empty((LOCAL_EXPERTS,), device=device, dtype=torch.int32)
    slot_row = torch.empty((n_slots,), device=device, dtype=torch.int32)
    a = torch.empty((max_rows, size_k), device=device, dtype=x.dtype)
    _count_kernel[(triton.cdiv(n_slots, 4096),)](local_ids, counters[0], n_slots, COUNT=LOCAL_EXPERTS, BLOCK=4096,
                                                num_warps=8)
    _scatter_kernel[(num_tokens,)](x, local_ids, counters[0], counters[1], a, slot_row, offs, D=size_k, TOPK=TOP_K,
                                   COUNT=LOCAL_EXPERTS, ALIGN=GROUP_ALIGN, num_warps=4)
    h = torch._grouped_mm(a, g["w13_bf16"].transpose(1, 2), offs=offs)
    act = torch.empty((max_rows, size_n), device=device, dtype=x.dtype)
    _silu_mul_kernel[(max_rows,)](h, offs, g["w13_gs"], act, I=size_n, COUNT=LOCAL_EXPERTS, BLOCK=size_n,
                                  num_warps=8)
    y = torch._grouped_mm(act, g["w2_bf16"].transpose(1, 2), offs=offs)
    out = torch.empty((num_tokens, size_k), device=device, dtype=x.dtype)
    _grouped_finalize_kernel[(num_tokens,)](y, local_ids, slot_row, topk_weights.float(), g["w2_gs"],
                                            g["post_expert_raw"], out, H=size_k, TOPK=TOP_K, COUNT=LOCAL_EXPERTS,
                                            EPS=POST_NORM_EPS, num_warps=8)
    return out


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    if num_tokens >= GROUPED_MIN_TOKENS:
        return _grouped_experts(x, topk_weights.contiguous(), bank["expert_map"][topk_ids].contiguous(), state)
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


def _fa3_attention(qkv, data, cfg, state):
    """SGLang's attention path (QK-norm, cache store, FA3, gated norm), for blocks longer than 8 tokens."""
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8), k_cache=k_cache, v_cache=v_cache, page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    return state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    dense = state["dense"]
    grouped = tokens >= GROUPED_MIN_TOKENS
    main_stream = torch.cuda.current_stream()
    if grouped:
        # The memory-bound expert dequant runs beside the compute-bound attention and projections.
        dequant_stream = state["dequant_stream"]
        dequant_stream.wait_stream(main_stream)
        with torch.cuda.stream(dequant_stream):
            _dequant_experts(state)
    # Attention: MuseSpark1xAttention.forward.
    qkv = _proj(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], state)
    if length <= 8:
        # QK-norm, RoPE, the FP8 cache write, attention and the gated output norm: muse_attention_decode's kernels.
        attn_cfg = {key: cfg[key] for key in ("batch", "q_len", "num_q_heads", "num_kv_heads", "head_dim", "page_size",
                                             "window", "rope_theta", "qk_scale_factor", "sm_scale", "kv_scale")}
        o = attn_fn({"qkv": qkv.view(batch, length, -1), "positions": data["positions"], "cache_loc": data["cache_loc"],
                     "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
                     "scheduler_metadata": data["scheduler_metadata"], "k_cache": data["k_cache"],
                     "v_cache": data["v_cache"], "k_scale": data["k_scale"], "v_scale": data["v_scale"],
                     "page_table": data["page_table"], "seq_lens": data["seq_lens"], "config": attn_cfg}).view(tokens, hq * d)

    else:
        o = _fa3_attention(qkv, data, cfg, state)
    attn_output = _proj(o, "o", HIDDEN, state)  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    side_stream = state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _proj(ffn_input, "pooler", D_EXPERT, state)
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    if grouped:
        main_stream.wait_stream(dequant_stream)
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _proj(expert_output, "unpooler", UNPOOL_SLICE, state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _proj(expert_output, "unpooler", HIDDEN, state)

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
