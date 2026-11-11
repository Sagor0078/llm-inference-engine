# okc: problem=muse_bf16_proj
"""Probe 22, not a submission: native mma.sync skinny kernel v2 (skinny_mma.cu: K-permuted
A-fragment repack, 16-byte x fragments, warps split K with smem atomics, cluster/DSMEM split-K) for m = 8/16/32 against v5's
kernels, as cold-L2 CUDA-graph replays. The v5 submission is inlined below."""
"""Skinny projections as split-K Triton kernels.

Every scored shape is bandwidth-bound (m <= 256), so the kernels are built to stream W once at
full HBM rate: (n-tile, k-slice) work units sized to fill the SMs in one wave, partial sums
reduced in-kernel by fp32 atomics into a persistent zeroed workspace, and the last unit to
finish a tile writes the output and re-zeroes the workspace, so a call is a single launch.

BF16 layers (m <= 32) load through TMA with the weights as the wide MMA operand ("swap AB":
out^T = W @ x^T), so a 64-row weight tile feeds the tensor cores against all of m at once.

The router's FP32 weights run as a BF16 hi + lo pair (split once in prepare, as SGLang does
above 8 tokens) with both halves through the tensor cores into one FP32 accumulator, in one
pass over the weights instead of two GEMMs and an add, at every m.

Above 32 tokens the BF16 layers stay on cuBLAS, which nothing here beat there, except the
pooler at 65-128 tokens (tune3.py, tune5.py).
Configs come from sweeping CUDA-graph replays from a cold L2 on the judge's H200.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


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


def _sub_prepare(weights: Weights):
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
            kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
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


import time as _time

MMA_CU = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
// 2 <= m <= 32 projections: out[m, N] = x[m, K] @ W[N, K]^T on mma.sync m16n8k16 with W as the
// A operand. W is repacked once (prepare) into A-fragment order, [N/16][K/16][32 lanes][8 bf16],
// after permuting K within each 32-wide block so that a lane's B fragments for two consecutive
// k16 chunks are one contiguous 16-byte piece of an x row (the dot product is order-invariant;
// x is read in place with the same permutation). A warp's A fragment for one 16x16 tile is then
// one coalesced 16-byte load per lane straight from HBM into registers, double-buffered.
// A CTA owns R row tiles; its WARPS warps split the CTA's K-range and reduce with shared-memory
// atomics; the CL CTAs of a cluster split K further and reduce through distributed shared
// memory, so no partial sum goes through global memory.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// Wp: repacked W. x [m, K] row-major bf16. out [m, N] bf16. U (chunks per batch) is even.
// The host guarantees (K / 16) % (CL * WARPS * U) == 0 and N % (R * 16) == 0.
template <int WARPS, int R, int NT, int U, int CL>
__global__ void __launch_bounds__(WARPS * 32) skinny_mma_kernel(const uint4* __restrict__ Wp,
                                                                const __nv_bfloat16* __restrict__ x,
                                                                __nv_bfloat16* __restrict__ out, int m, int N, int K) {
  constexpr int ROWS = R * 16;
  constexpr int TOK = NT * 8;
  constexpr int P = U / 2;  // chunk pairs per batch
  __shared__ float red[ROWS * TOK];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int rank = CL > 1 ? (int)cg::this_cluster().block_rank() : 0;
  const int group = blockIdx.x / CL;
  const int C = K / 16;
  const int cw = C / CL / WARPS;  // chunks per warp
  const int c0 = (rank * WARPS + warp) * cw;
  const int nb = cw / U;

  const uint4* wp[R];
#pragma unroll
  for (int r = 0; r < R; ++r) wp[r] = Wp + ((size_t)(group * R + r) * C + c0) * 32 + lane;
  const uint4* xq[NT];
  bool tv[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int tok = j * 8 + g;
    tv[j] = tok < m;
    xq[j] = reinterpret_cast<const uint4*>(x + (size_t)(tv[j] ? tok : 0) * K + c0 * 16) + t;
  }

  uint4 a[U][R], an[U][R];
  uint4 b[P][NT], bn[P][NT];
  auto load = [&](uint4 (&aa)[U][R], uint4 (&bb)[P][NT], int c) {
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < R; ++r) aa[u][r] = ld_stream(wp[r] + (size_t)(c + u) * 32);
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
      for (int j = 0; j < NT; ++j) bb[p][j] = tv[j] ? __ldg(xq[j] + (c / 2 + p) * 4) : make_uint4(0, 0, 0, 0);
  };
  load(a, b, 0);
  for (int i = threadIdx.x; i < ROWS * TOK; i += WARPS * 32) red[i] = 0.f;
  __syncthreads();

  float acc[R][NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.f;

  for (int i = 0; i < nb; ++i) {
    if (i + 1 < nb) load(an, bn, (i + 1) * U);
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint4& bv = b[u / 2][j];
        const uint32_t b0 = (u & 1) ? bv.z : bv.x, b1 = (u & 1) ? bv.w : bv.y;
#pragma unroll
        for (int r = 0; r < R; ++r) mma_bf16(acc[r][j], a[u][r], b0, b1);
      }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < R; ++r) a[u][r] = an[u][r];
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
      for (int j = 0; j < NT; ++j) b[p][j] = bn[p][j];
  }

  // c0, c1: (row g, tokens 2t, 2t + 1); c2, c3: (row g + 8, same tokens).
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      float* r0 = red + (r * 16 + g) * TOK + j * 8 + 2 * t;
      atomicAdd(r0, acc[r][j][0]);
      atomicAdd(r0 + 1, acc[r][j][1]);
      atomicAdd(r0 + 8 * TOK, acc[r][j][2]);
      atomicAdd(r0 + 8 * TOK + 1, acc[r][j][3]);
    }
  constexpr int RPR = ROWS / CL;  // rows this CTA reduces and writes
  if (CL > 1) cg::this_cluster().sync();
  else __syncthreads();
  for (int i = threadIdx.x; i < RPR * TOK; i += WARPS * 32) {
    const int tok = i / RPR, row = rank * RPR + i % RPR;
    if (tok < m) {
      const int off = row * TOK + tok;
      float s = red[off];
      if (CL > 1) {
        auto cluster = cg::this_cluster();
#pragma unroll
        for (int q = 1; q < CL; ++q) s += cluster.map_shared_rank(red, (rank + q) % CL)[off];
      }
      out[(size_t)tok * N + group * ROWS + row] = __float2bfloat16_rn(s);
    }
  }
  if (CL > 1) cg::this_cluster().sync();  // keep this CTA's partials alive until every peer has read them
}

template <int WARPS, int R, int NT, int U, int CL>
static void launch_t(const void* Wp, const void* x, void* out, int m, int N, int K, cudaStream_t s) {
  auto kern = skinny_mma_kernel<WARPS, R, NT, U, CL>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / (R * 16)) * CL);
  cfg.blockDim = dim3(WARPS * 32);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CL;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<const uint4*>(Wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<__nv_bfloat16*>(out), m, N, K);
}

#define SKINNY_CL(W, R, NT, U)                                                   \
  switch (cl) {                                                                  \
    case 1: launch_t<W, R, NT, U, 1>(Wp, x, out, m, N, K, s); break;             \
    case 2: launch_t<W, R, NT, U, 2>(Wp, x, out, m, N, K, s); break;             \
    case 4: launch_t<W, R, NT, U, 4>(Wp, x, out, m, N, K, s); break;             \
    default: launch_t<W, R, NT, U, 8>(Wp, x, out, m, N, K, s); break;            \
  }

template <int WARPS, int R, int NT>
static void launch_u(const void* Wp, const void* x, void* out, int m, int N, int K, int u, int cl, cudaStream_t s) {
  if (u == 2) { SKINNY_CL(WARPS, R, NT, 2) }
  else { SKINNY_CL(WARPS, R, NT, 4) }
}

template <int WARPS, int R>
static void launch_n(const void* Wp, const void* x, void* out, int m, int N, int K, int u, int cl, cudaStream_t s) {
  if (m <= 8) launch_u<WARPS, R, 1>(Wp, x, out, m, N, K, u, cl, s);
  else if (m <= 16) launch_u<WARPS, R, 2>(Wp, x, out, m, N, K, u, cl, s);
  else launch_u<WARPS, R, 4>(Wp, x, out, m, N, K, u, cl, s);
}

template <int WARPS>
static void launch_r(const void* Wp, const void* x, void* out, int m, int N, int K, int r, int u, int cl,
                     cudaStream_t s) {
  if (r == 1) launch_n<WARPS, 1>(Wp, x, out, m, N, K, u, cl, s);
  else if (r == 2) launch_n<WARPS, 2>(Wp, x, out, m, N, K, u, cl, s);
  else launch_n<WARPS, 4>(Wp, x, out, m, N, K, u, cl, s);
}

void skinny_mma_launch(const void* Wp, const void* x, void* out, int m, int N, int K, int warps, int r, int u,
                       int cl, cudaStream_t s) {
  if (warps == 4) launch_r<4>(Wp, x, out, m, N, K, r, u, cl, s);
  else launch_r<8>(Wp, x, out, m, N, K, r, u, cl, s);
}

void skinny_mma(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, int64_t n, int64_t warps, int64_t r, int64_t u, int64_t cl) {
  skinny_mma_launch(Wp.data_ptr(), x.data_ptr(), out.data_ptr(), (int)x.size(0), (int)n, (int)x.size(1), (int)warps,
                    (int)r, (int)u, (int)cl, at::cuda::getCurrentCUDAStream());
}
"""
MMA_CPP = "void skinny_mma(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, int64_t n, int64_t warps, int64_t r, int64_t u, int64_t cl);"


def _kperm(k, device):
    """Within each 32-wide block of K, the order that puts a lane's B fragments for two k16 chunks
    in one contiguous 16-byte piece of x: logical (half h, position p) -> physical index."""
    L = torch.arange(32)
    h, p = L // 16, L % 16
    phys = ((p % 8) // 2) * 8 + h * 4 + (p // 8) * 2 + p % 2
    return (torch.arange(k // 32)[:, None] * 32 + phys[None, :]).flatten().to(device)


def _repack(w):
    n, k = w.shape
    w = w[:, _kperm(k, w.device)]
    return w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()


def prepare(weights):
    t0 = _time.time()
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="okc_mma_p22", cpp_sources=MMA_CPP, cuda_sources=MMA_CU, functions=["skinny_mma"],
                      extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    lines = [f"compile {_time.time() - t0:.0f}s"]
    state = _sub_prepare(weights)
    dev = state["router"].device
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)

    def bench(fn, reps=40):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ts = []
        for _ in range(reps):
            torch.sum(flush, dim=0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    nop = torch.zeros(1, device=dev)
    lines.append(f"nop {bench(lambda: nop.add_(1)):.1f}")
    packed = {l: _repack(state[l]) for l in ("pooler", "qkv_gate", "o", "unpooler")}
    plan = {  # (warps, R, U, CL)
        "pooler": [(8, 4, 2, 2), (8, 4, 4, 1), (8, 2, 4, 2), (8, 2, 2, 4), (4, 4, 4, 2), (8, 1, 4, 2), (4, 2, 4, 4)],
        "qkv_gate": [(8, 4, 2, 2), (8, 2, 4, 2), (8, 2, 2, 4), (4, 4, 4, 2), (8, 1, 4, 2), (4, 2, 4, 4), (8, 4, 4, 1)],
        "o": [(4, 4, 2, 2), (8, 4, 2, 1), (4, 2, 4, 1), (8, 2, 2, 2), (4, 4, 4, 1), (8, 2, 4, 1)],
        "unpooler_slice": [(8, 2, 2, 4), (8, 1, 2, 4), (8, 1, 2, 8), (4, 1, 4, 8), (8, 2, 4, 2), (4, 2, 2, 8)],
        "unpooler": [(8, 4, 4, 1), (8, 4, 2, 2), (4, 4, 4, 2), (8, 2, 4, 1), (8, 2, 2, 2), (4, 2, 4, 2)],
    }
    for m in (8, 16, 32):
        for key, cfgs in plan.items():
            if (key == "unpooler_slice") != (m <= 8) or (key == "unpooler" and m <= 8):
                continue
            if _time.time() - t0 > 400:
                lines.append("BUDGET")
                break
            layer = key.split("_slice")[0]
            w = state[layer]
            n = 1024 if key == "unpooler_slice" else w.shape[0]
            k = w.shape[1]
            x = torch.randn((m, k), device=dev).to(torch.bfloat16)
            ref = x.double() @ w[:n].double().T
            data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
            t_v5 = bench(lambda: kernel_fn(data, state))
            res = []
            for wa, rr, u, cl in cfgs:
                if rr == 4 and m > 16 and u == 4:
                    continue  # spills
                out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
                run = lambda: mod.skinny_mma(packed[layer], x, out, n, wa, rr, u, cl)
                try:
                    run()
                    torch.cuda.synchronize()
                    err = float((out.double() - ref).norm() / ref.norm())
                    res.append((bench(run), f"w{wa}r{rr}u{u}c{cl} e{err:.0e}"))
                except Exception as e:  # noqa: BLE001
                    res.append((999.0, f"w{wa}r{rr}u{u}c{cl} FAIL {type(e).__name__}: {str(e)[:120]}"))
            res.sort()
            lines.append(f"{key} m{m} v5 {t_v5:.1f} | " + " ; ".join(f"{t:.1f}/{s}" for t, s in res))
            print(lines[-1], flush=True)
    # odd m correctness on the first config of each layer
    for m in (2, 5, 7, 13, 27):
        for layer in ("pooler", "o"):
            w = state[layer]
            n, k = w.shape
            x = torch.randn((m, k), device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
            wa, rr, u, cl = plan[layer][0]
            mod.skinny_mma(packed[layer], x, out, n, wa, rr, u, cl)
            ref = x.double() @ w.double().T
            lines.append(f"odd m{m} {layer} err {float((out.double() - ref).norm() / ref.norm()):.1e}")
    lines.append(f"total {_time.time() - t0:.0f}s")
    raise RuntimeError("PROBE\n" + "\n".join(lines))
