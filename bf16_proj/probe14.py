# okc: problem=muse_bf16_proj
"""Probe 14 (skinny kernel v3, 4-stage cp.async pipeline), not a submission: the native CUDA skinny kernel (lane-packed weights, mma.m16n8k16, warp
+ cluster/DSMEM reduction) against v5's own path, same process, CUDA-graph replays from a cold L2."""

import time
from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

CUDA_SRC = r"""// Skinny BF16 projections (m <= 32): out[m, n] = x[m, k] @ W[n, k]^T, FP32 accumulation.
//
// W arrives pre-packed (lossless reorder, in prepare) as [n/16 stripes][k/16 steps][32 lanes][8 bf16]:
// lane l's 16 bytes are exactly its A fragment of mma.m16n8k16 for that (stripe, step), and a stripe's
// consecutive steps are one contiguous run.
//
// A CTA owns R stripes (16R rows) and one rank's share of the k steps, walked in stages of SC = WARPS
// steps through a STAGES-deep cp.async ring in shared memory: per stage, R contiguous runs of W
// (SC * 512 bytes each) and the matching x tile (tokens past m zero-filled by the copy itself). While
// one stage is computed the next STAGES - 1 are in flight. Warp w computes step w of each stage for all
// R stripes, reading its A fragment with one 128-bit shared load and x as the B operand.
//
// Warp partials meet in shared memory (conflict-free atomics, C-fragment order); when a stripe group is
// split over SPLIT CTAs those CTAs form one thread-block cluster and rank 0 sums the others' partials
// through distributed shared memory: no global atomics, no arrival counter, no workspace that outlives
// a call.
//
// HI_LO (the router): W is the FP32 weight as a packed BF16 hi + lo pair, two MMAs per fragment into one
// FP32 accumulator; the output is FP32.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

constexpr int WARPS = 8;
constexpr int SC = WARPS;              // k steps (of 16) per stage: one per warp
constexpr int STAGES = 4;
constexpr int XS_STRIDE = SC * 16 + 8;  // bf16 per staged x row: +8 keeps B-fragment reads conflict-free

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

template <int NT, bool HI_LO, int R>
struct Smem {
  static constexpr int W_ARRAYS = HI_LO ? 2 : 1;
  static constexpr int W_BYTES = W_ARRAYS * R * SC * 512;  // per stage
  static constexpr int X_BYTES = 8 * NT * XS_STRIDE * 2;   // per stage
  static constexpr int STAGE_BYTES = W_BYTES + X_BYTES;
  static constexpr int PART_BYTES = R * NT * 4 * 32 * 4;
  static constexpr int TOTAL = STAGES * STAGE_BYTES + PART_BYTES;
};

// NT n-tiles of 8 tokens (m <= 8 NT); R stripes per CTA.
template <int NT, bool HI_LO, int R>
__global__ void __launch_bounds__(WARPS * 32) skinny_kernel(const uint4* __restrict__ w, const uint4* __restrict__ w_lo,
                                                            const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
                                                            int m, int n, int k, int split) {
  using L = Smem<NT, HI_LO, R>;
  extern __shared__ __align__(16) unsigned char smem[];
  float* part = reinterpret_cast<float*>(smem + STAGES * L::STAGE_BYTES);  // [R][NT][4][32]

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int stripe0 = blockIdx.x * R;
  const int rank = blockIdx.y;
  const int ksteps = k >> 4;
  const int nstages_all = ksteps / SC;
  const int c0 = (int)((long long)rank * nstages_all / split);
  const int c1 = (int)((long long)(rank + 1) * nstages_all / split);
  const int nst = c1 - c0;
  const int g = lane >> 2, t = lane & 3;

  for (int i = threadIdx.x; i < R * NT * 128; i += WARPS * 32) part[i] = 0.f;

  // Issue the copies of stage c (global stage index c0 + c) into ring slot c % STAGES.
  auto issue = [&](int c) {
    unsigned char* base = smem + (c % STAGES) * L::STAGE_BYTES;
    const int st0 = (c0 + c) * SC;
    // W: W_ARRAYS x R runs of SC * 512 bytes = SC * 32 pieces of 16 bytes each.
    constexpr int W_PIECES = L::W_ARRAYS * R * SC * 32;
    for (int p = threadIdx.x; p < W_PIECES; p += WARPS * 32) {
      const int arr = p / (R * SC * 32), q = p % (R * SC * 32);
      const int r = q / (SC * 32), o = q % (SC * 32);
      const uint4* src = (arr ? w_lo : w) + ((size_t)(stripe0 + r) * ksteps + st0) * 32 + o;
      cp_async16(base + (size_t)p * 16, src, true);
    }
    // x tile: 8 NT rows x SC * 16 bf16 = SC * 2 pieces per row; rows past m zero-filled.
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(base + L::W_BYTES);
    for (int p = threadIdx.x; p < 8 * NT * SC * 2; p += WARPS * 32) {
      const int tok = p / (SC * 2), c8 = p % (SC * 2);
      const bool live = tok < m;
      const __nv_bfloat16* src = x + (size_t)(live ? tok : 0) * k + (size_t)st0 * 16 + c8 * 8;
      cp_async16(xs + tok * XS_STRIDE + c8 * 8, src, live);
    }
  };

  float acc[R][NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.f;

#pragma unroll
  for (int c = 0; c < STAGES - 1; ++c) {
    if (c < nst) issue(c);
    cp_async_commit();  // empty groups keep the wait count uniform
  }
  for (int c = 0; c < nst; ++c) {
    cp_async_wait<STAGES - 2>();  // stage c has landed (this thread's copies)
    __syncthreads();              // ...and everyone's; slot (c - 1) % STAGES is free again
    if (c + STAGES - 1 < nst) issue(c + STAGES - 1);
    cp_async_commit();
    const unsigned char* base = smem + (c % STAGES) * L::STAGE_BYTES;
    const uint4* wt = reinterpret_cast<const uint4*>(base);
    const __nv_bfloat16* xs = reinterpret_cast<const __nv_bfloat16*>(base + L::W_BYTES);
    const int kc = warp * 16 + 2 * t;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xs + (j * 8 + g) * XS_STRIDE + kc);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xs + (j * 8 + g) * XS_STRIDE + kc + 8);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        mma_bf16(acc[r][j], wt[(r * SC + warp) * 32 + lane], b0, b1);
        if (HI_LO) mma_bf16(acc[r][j], wt[((R + r) * SC + warp) * 32 + lane], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  // C fragment: acc[r][j][0..1] = (row g, tokens 8j + 2t, +1), [2..3] = (row g + 8, same tokens).
  __syncthreads();  // part zeroed by every thread before any adds (nst may be 0)
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) atomicAdd(&part[((r * NT + j) * 4 + e) * 32 + lane], acc[r][j][e]);

  const int row0 = stripe0 * 16;
  if (split > 1) {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();  // every rank's partial is complete in its shared memory
    if (rank == 0) {
      for (int i = threadIdx.x; i < R * NT * 128; i += WARPS * 32) {
        float v = part[i];
        for (int q = 1; q < split; ++q) v += cluster.map_shared_rank(part, q)[i];
        part[i] = v;
      }
    }
    cluster.sync();  // the others' shared memory stays alive until rank 0 has read it
    if (rank != 0) return;
  } else {
    __syncthreads();
  }
  for (int i = threadIdx.x; i < R * NT * 128; i += WARPS * 32) {
    const int r = i / (NT * 128), j = (i / 128) % NT, e = (i / 32) & 3, l = i & 31;
    const int row = r * 16 + (l >> 2) + ((e & 2) ? 8 : 0);
    const int tok = j * 8 + (l & 3) * 2 + (e & 1);
    if (tok < m) {
      if (HI_LO)
        reinterpret_cast<float*>(out)[(size_t)tok * n + row0 + row] = part[i];
      else
        reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row0 + row] = __float2bfloat16_rn(part[i]);
    }
  }
}

template <int NT, bool HI_LO, int R>
static cudaError_t launch_t(const void* w, const void* w_lo, const void* x, void* out, int m, int n, int k, int split,
                            cudaStream_t stream) {
  auto kern = skinny_kernel<NT, HI_LO, R>;
  constexpr int smem = Smem<NT, HI_LO, R>::TOTAL;
  cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return e;
  if (split > 8) {
    e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return e;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(n / (16 * R), split, 1);
  cfg.blockDim = dim3(WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = split;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = split > 1 ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kern, static_cast<const uint4*>(w), static_cast<const uint4*>(w_lo),
                            static_cast<const __nv_bfloat16*>(x), out, m, n, k, split);
}

// R = 4 stripes (64 rows) per CTA for BF16 layers, 2 for the router (two weight arrays per stage).
// Returns a cudaError_t code (0 on success). n must be a multiple of 16 R, k of 16 SC.
int skinny_launch(const void* w, const void* w_lo, const void* x, void* out, int m, int n, int k, int split, bool hi_lo,
                  cudaStream_t stream) {
  const int nt = m <= 8 ? 1 : m <= 16 ? 2 : 4;
  cudaError_t e;
  if (hi_lo) {
    e = nt == 1 ? launch_t<1, true, 2>(w, w_lo, x, out, m, n, k, split, stream)
        : nt == 2 ? launch_t<2, true, 2>(w, w_lo, x, out, m, n, k, split, stream)
                  : launch_t<4, true, 2>(w, w_lo, x, out, m, n, k, split, stream);
  } else {
    e = nt == 1 ? launch_t<1, false, 4>(w, w_lo, x, out, m, n, k, split, stream)
        : nt == 2 ? launch_t<2, false, 4>(w, w_lo, x, out, m, n, k, split, stream)
                  : launch_t<4, false, 4>(w, w_lo, x, out, m, n, k, split, stream);
  }
  return (int)e;
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
void skinny_run(torch::Tensor w, torch::Tensor w_lo, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n,
                int64_t k, int64_t split, bool hi_lo) {
  int e = skinny_launch(w.data_ptr(), w_lo.data_ptr(), x.data_ptr(), out.data_ptr(), (int)m, (int)n, (int)k, (int)split,
                        hi_lo, c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "skinny_launch: ", cudaGetErrorString((cudaError_t)e));
}
"""
CPP_SRC = "void skinny_run(torch::Tensor w, torch::Tensor w_lo, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n, int64_t k, int64_t split, bool hi_lo);"


class Weights(TypedDict):
    router: torch.Tensor


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


def v5_prepare(weights):
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
            v5_kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return state


def v5_kernel_fn(data, state) -> torch.Tensor:
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



def _pack(w):
    n, k = w.shape
    v = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2)
    return v.permute(0, 3, 2, 5, 4, 1, 6).contiguous()


def prepare(weights):
    t0 = time.time()
    lines = []
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="okc_skinny_p14", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["skinny_run"],
                      extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    lines.append(f"compile {time.time() - t0:.1f}s")
    v5_state = v5_prepare(weights)
    lines.append(f"v5 prepare done [{time.time() - t0:.0f}s]")
    dev = weights["router"].device
    packed = {name: _pack(w.contiguous()) for name, w in weights.items() if name != "router"}
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    packed["router"] = _pack(hi)
    packed_lo = _pack(lo)
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
    lines.append(f"nop {bench(lambda: nop.add_(1)):.2f}")
    shapes = [(1, "unpooler", (4, 8, 16)), (8, "unpooler", (4, 8, 16)), (1, "router", (8, 16)), (16, "router", (8, 16)),
              (32, "router", (8, 16)), (1, "o", (1, 2)), (32, "o", (1, 2)), (1, "qkv_gate", (2, 4, 8)), (8, "qkv_gate", (2, 4, 8)),
              (16, "qkv_gate", (2, 4, 8)), (32, "qkv_gate", (2, 4, 8)), (1, "pooler", (1, 2, 4)), (16, "pooler", (1, 2, 4)),
              (32, "pooler", (1, 2, 4)), (16, "unpooler", (1, 2)), (32, "unpooler", (1, 2)), (5, "qkv_gate", (4,)), (27, "router", (16,))]
    for m, layer, splits in shapes:
        n, k = weights[layer].shape
        if layer == "unpooler" and m <= 8:
            n = 1024
        is_r = layer == "router"
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
        ref = x.double() @ weights[layer][:n].double().T
        tol = 1e-4 if is_r else 1e-2
        v5_out = v5_kernel_fn(data, v5_state)
        v5_err = float((v5_out.double() - ref).norm() / ref.norm())
        v5_t = bench(lambda: v5_kernel_fn(data, v5_state))
        row = [f"v5={v5_t:.2f}" + ("" if v5_err < tol else f"!BAD{v5_err:.0e}")]
        w = packed[layer]
        wl = packed_lo if is_r else w
        out = torch.empty((m, n), device=dev, dtype=torch.float32 if is_r else torch.bfloat16)
        for split in splits:
            try:
                fn = lambda split=split: mod.skinny_run(w, wl, x, out, m, n, k, split, is_r)
                out.fill_(float("nan"))
                fn()
                torch.cuda.synchronize()
                err = float((out.double() - ref).norm() / ref.norm())
                t = bench(fn)
                out.fill_(float("nan"))
                fn()
                torch.cuda.synchronize()
                err2 = float((out.double() - ref).norm() / ref.norm())
                ok = err < tol and err2 < tol
                row.append(f"cu_s{split}={t:.2f}" + ("" if ok else f"!BAD{err:.0e}/{err2:.0e}"))
            except Exception as e:  # noqa: BLE001
                row.append(f"cu_s{split}=EXC:{type(e).__name__}:{str(e)[:80]}")
        lines.append(f"{layer[:4]} m{m} n{n} | " + " ".join(row) + f" [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
