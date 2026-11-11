# okc: problem=muse_bf16_proj
"""Probe 11, not a submission: the native CUDA skinny kernel (lane-packed weights, mma.m16n8k16, warp
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
// lane l's 16 bytes are exactly its A fragment of mma.m16n8k16 for that (stripe, step), so each warp
// streams 512 contiguous bytes per step with one 128-bit load per lane and no shared-memory staging.
// x is the B operand (tokens as the MMA's n), read straight from global (it is small and L1/L2-hot);
// tokens past m read as zero.
//
// Each CTA owns one 16-row stripe and a slice of the k steps; its warps interleave over the steps and
// are reduced through shared memory. When a stripe is split over SPLIT CTAs, those CTAs form one
// thread-block cluster and rank 0 sums the others' partials through distributed shared memory, so
// there are no global atomics, no arrival counter and no workspace that outlives a call.
//
// HI_LO (the router): W is the FP32 weight as a packed BF16 hi + lo pair, two MMAs per step into
// one FP32 accumulator; the output is FP32.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

constexpr int WARPS = 8;
constexpr int UNROLL = 4;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// NT n-tiles of 8 tokens (m <= 8 * NT).
template <int NT, bool HI_LO>
__global__ void __launch_bounds__(WARPS * 32) skinny_kernel(const uint4* __restrict__ w, const uint4* __restrict__ w_lo,
                                                            const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
                                                            int m, int n, int k, int split) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int stripe = blockIdx.x;
  const int rank = blockIdx.y;  // rank within the cluster when split > 1
  const int ksteps = k >> 4;
  const int s0 = (int)((long long)rank * ksteps / split);
  const int s1 = (int)((long long)(rank + 1) * ksteps / split);
  const int g = lane >> 2, t = lane & 3;

  // This lane's B rows (tokens) and whether they exist; x row stride is k.
  const uint32_t* xr[NT];
  bool live[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int tok = j * 8 + g;
    live[j] = tok < m;
    xr[j] = reinterpret_cast<const uint32_t*>(x + (size_t)(live[j] ? tok : 0) * k) + t;
  }

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  const uint4* wp = w + ((size_t)stripe * ksteps) * 32 + lane;
  const uint4* wlp = HI_LO ? w_lo + ((size_t)stripe * ksteps) * 32 + lane : nullptr;

  for (int s = s0 + warp; s < s1; s += WARPS * UNROLL) {
    uint4 a[UNROLL], al[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int st = s + u * WARPS;
      if (st < s1) {
        a[u] = ld_stream(wp + (size_t)st * 32);
        if (HI_LO) al[u] = ld_stream(wlp + (size_t)st * 32);
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int st = s + u * WARPS;
      if (st < s1) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          // B fragment: x[tok][16 st + 2t .. +1] and x[tok][16 st + 2t + 8 .. +9], as bf16 pairs.
          const uint32_t b0 = live[j] ? __ldg(xr[j] + st * 8) : 0u;
          const uint32_t b1 = live[j] ? __ldg(xr[j] + st * 8 + 4) : 0u;
          mma_bf16(acc[j], a[u], b0, b1);
          if (HI_LO) mma_bf16(acc[j], al[u], b0, b1);
        }
      }
    }
  }

  // C fragment: acc[j][0..1] = (row g, tokens 8j + 2t, +1), acc[j][2..3] = (row g + 8, same tokens).
  // Reduce over warps through shared memory: red[warp][j][e][lane].
  __shared__ float red[WARPS][NT][4][32];
  __shared__ float part[16 * 8 * NT];  // this CTA's [row 0..15][token 0..8NT) partial
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) red[warp][j][e][lane] = acc[j][e];
  __syncthreads();
  for (int i = threadIdx.x; i < NT * 4 * 32; i += WARPS * 32) {
    const int j = i / 128, e = (i / 32) & 3, l = i & 31;
    float v = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < WARPS; ++w8) v += red[w8][j][e][l];
    const int row = (l >> 2) + ((e & 2) ? 8 : 0);
    const int tok = j * 8 + (l & 3) * 2 + (e & 1);
    part[row * 8 * NT + tok] = v;
  }

  const int row0 = stripe * 16;
  if (split > 1) {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();  // every rank's partial is in its shared memory
    if (rank == 0) {
      for (int i = threadIdx.x; i < 16 * 8 * NT; i += WARPS * 32) {
        float v = part[i];
        for (int r = 1; r < split; ++r) v += cluster.map_shared_rank(part, r)[i];
        part[i] = v;
      }
    }
    cluster.sync();  // the others' shared memory stays alive until rank 0 has read it
    if (rank != 0) return;
  } else {
    __syncthreads();
  }
  for (int i = threadIdx.x; i < 16 * 8 * NT; i += WARPS * 32) {
    const int row = i / (8 * NT), tok = i % (8 * NT);
    if (tok < m) {
      if (HI_LO)
        reinterpret_cast<float*>(out)[(size_t)tok * n + row0 + row] = part[i];
      else
        reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row0 + row] = __float2bfloat16_rn(part[i]);
    }
  }
}

template <int NT, bool HI_LO>
static cudaError_t launch_t(const void* w, const void* w_lo, const void* x, void* out, int m, int n, int k, int split,
                            cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(n / 16, split, 1);
  cfg.blockDim = dim3(WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = split;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = split > 1 ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, skinny_kernel<NT, HI_LO>, static_cast<const uint4*>(w),
                            static_cast<const uint4*>(w_lo), static_cast<const __nv_bfloat16*>(x), out, m, n, k, split);
}

// Returns a cudaError_t code (0 on success).
int skinny_launch(const void* w, const void* w_lo, const void* x, void* out, int m, int n, int k, int split, bool hi_lo,
                  cudaStream_t stream) {
  const int nt = m <= 8 ? 1 : m <= 16 ? 2 : 4;
  cudaError_t e;
  if (hi_lo) {
    e = nt == 1 ? launch_t<1, true>(w, w_lo, x, out, m, n, k, split, stream)
        : nt == 2 ? launch_t<2, true>(w, w_lo, x, out, m, n, k, split, stream)
                  : launch_t<4, true>(w, w_lo, x, out, m, n, k, split, stream);
  } else {
    e = nt == 1 ? launch_t<1, false>(w, w_lo, x, out, m, n, k, split, stream)
        : nt == 2 ? launch_t<2, false>(w, w_lo, x, out, m, n, k, split, stream)
                  : launch_t<4, false>(w, w_lo, x, out, m, n, k, split, stream);
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
    mod = load_inline(name="okc_skinny_p11b", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["skinny_run"],
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
    shapes = [(1, "unpooler", (2, 4, 8)), (8, "unpooler", (2, 4, 8)), (1, "router", (4, 8)), (16, "router", (4, 8)),
              (32, "router", (4, 8)), (1, "o", (1, 2)), (32, "o", (1, 2)), (1, "qkv_gate", (1, 2, 4)), (8, "qkv_gate", (1, 2, 4)),
              (16, "qkv_gate", (1, 2, 4)), (32, "qkv_gate", (1, 2, 4)), (1, "pooler", (1, 2)), (16, "pooler", (1, 2)),
              (32, "pooler", (1, 2)), (16, "unpooler", (1, 2)), (32, "unpooler", (1, 2)), (5, "qkv_gate", (2,)), (27, "router", (8,))]
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
