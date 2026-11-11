# okc: problem=muse_bf16_proj
"""Probe 20, not a submission: native CUDA streaming GEMV (gemv1.cu) for every m = 1 shape against
v5's kernels and cuBLAS, as cold-L2 CUDA-graph replays. The v5 submission is inlined below."""
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

GEMV_CU = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
// m = 1 projections as a streaming GEMV. Each warp owns one row's K-segment and WPR warps share a
// row, reducing in shared memory, so there is no split-K hand-off through global memory. x goes to
// shared memory once per CTA (cp.async, overlapped with the first weight batch); the weights are
// double-buffered in registers: batch i + 1 is in flight while batch i is multiplied.
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ float2 bf2_to_f2(uint32_t v) {
  return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}

template <int U>
__device__ __forceinline__ void load_batch(uint4 (&w)[U], const uint4* wp, int v0, int lane) {
#pragma unroll
  for (int u = 0; u < U; ++u) w[u] = ld_stream(wp + v0 + u * 32 + lane);
}

// W [N, K] (bf16, or fp32 when F32W), x [K] bf16, out [N] (bf16, or fp32 when F32W).
// The host guarantees (K / WPR / EPV) % (32 * U) == 0 and K % 8 == 0.
template <int WPR, int U, bool F32W>
__global__ void __launch_bounds__(256) gemv1_kernel(const void* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    void* __restrict__ out, int N, int K) {
  constexpr int RPC = 8 / WPR;      // rows per CTA
  constexpr int EPV = F32W ? 4 : 8; // weights per 16-byte vector
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
  float* part = reinterpret_cast<float*>(smem + K * 2);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = min(blockIdx.x * RPC + warp / WPR, N - 1);  // a clamped row recomputes N - 1
  const int seg = warp % WPR;
  const int ks = K / WPR;
  const int nb = ks / EPV / (32 * U);
  const uint4* wp = reinterpret_cast<const uint4*>(
      reinterpret_cast<const char*>(W) + ((size_t)row * K + (size_t)seg * ks) * (F32W ? 4 : 2));

  uint4 w[U], wn[U];
  load_batch<U>(w, wp, 0, lane);
  for (int i = threadIdx.x; i < K / 8; i += 256) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(xs + i * 8));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(x + i * 8));
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  const __nv_bfloat16* xseg = xs + seg * ks;
  float acc0 = 0.f, acc1 = 0.f;
  for (int b = 0; b < nb; ++b) {
    if (b + 1 < nb) load_batch<U>(wn, wp, (b + 1) * 32 * U, lane);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = b * 32 * U + u * 32 + lane;
      if (F32W) {
        const uint2 xv = *reinterpret_cast<const uint2*>(xseg + v * 4);
        const float2 x01 = bf2_to_f2(xv.x), x23 = bf2_to_f2(xv.y);
        acc0 = fmaf(__uint_as_float(w[u].x), x01.x, acc0);
        acc1 = fmaf(__uint_as_float(w[u].y), x01.y, acc1);
        acc0 = fmaf(__uint_as_float(w[u].z), x23.x, acc0);
        acc1 = fmaf(__uint_as_float(w[u].w), x23.y, acc1);
      } else {
        const uint4 xv = *reinterpret_cast<const uint4*>(xseg + v * 8);
        const uint32_t wa[4] = {w[u].x, w[u].y, w[u].z, w[u].w};
        const uint32_t xa[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 wf = bf2_to_f2(wa[j]), xf = bf2_to_f2(xa[j]);
          acc0 = fmaf(wf.x, xf.x, acc0);
          acc1 = fmaf(wf.y, xf.y, acc1);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) w[u] = wn[u];
  }
  float acc = acc0 + acc1;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (WPR > 1) {
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (lane == 0 && seg == 0) {
#pragma unroll
      for (int s = 1; s < WPR; ++s) acc += part[warp + s];
    }
  }
  if (lane == 0 && seg == 0) {
    if (F32W) reinterpret_cast<float*>(out)[row] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[row] = __float2bfloat16_rn(acc);
  }
}

template <int WPR, int U, bool F32W>
static void launch_t(const void* W, const void* x, void* out, int N, int K, cudaStream_t s) {
  constexpr int RPC = 8 / WPR;
  const int smem = K * 2 + 8 * 4;
  auto kern = gemv1_kernel<WPR, U, F32W>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kern<<<(N + RPC - 1) / RPC, 256, smem, s>>>(W, reinterpret_cast<const __nv_bfloat16*>(x), out, N, K);
}

template <int WPR, bool F32W>
static void launch_u(const void* W, const void* x, void* out, int N, int K, int u, cudaStream_t s) {
  if (u == 2) launch_t<WPR, 2, F32W>(W, x, out, N, K, s);
  else if (u == 4) launch_t<WPR, 4, F32W>(W, x, out, N, K, s);
  else launch_t<WPR, 8, F32W>(W, x, out, N, K, s);
}

template <bool F32W>
static void launch_w(const void* W, const void* x, void* out, int N, int K, int wpr, int u, cudaStream_t s) {
  if (wpr == 1) launch_u<1, F32W>(W, x, out, N, K, u, s);
  else if (wpr == 2) launch_u<2, F32W>(W, x, out, N, K, u, s);
  else if (wpr == 4) launch_u<4, F32W>(W, x, out, N, K, u, s);
  else launch_u<8, F32W>(W, x, out, N, K, u, s);
}

void gemv1_launch(const void* W, const void* x, void* out, int N, int K, int wpr, int u, bool f32w, cudaStream_t s) {
  if (f32w) launch_w<true>(W, x, out, N, K, wpr, u, s);
  else launch_w<false>(W, x, out, N, K, wpr, u, s);
}

void gemv1(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t n, int64_t wpr, int64_t u) {
  gemv1_launch(W.data_ptr(), x.data_ptr(), out.data_ptr(), (int)n, (int)x.size(-1), (int)wpr, (int)u,
               W.scalar_type() == at::kFloat, at::cuda::getCurrentCUDAStream());
}
"""
GEMV_CPP = "void gemv1(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t n, int64_t wpr, int64_t u);"


def prepare(weights):
    t0 = _time.time()
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="okc_gemv1_p20", cpp_sources=GEMV_CPP, cuda_sources=GEMV_CU, functions=["gemv1"],
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
    plan = {
        "pooler": [(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 4), (4, 8), (8, 4)],
        "qkv_gate": [(1, 4), (1, 8), (2, 4), (2, 8), (4, 4), (4, 8), (8, 2), (8, 4)],
        "o": [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1)],
        "unpooler": [(1, 4), (1, 8), (2, 4), (2, 8), (4, 2), (4, 4), (8, 2)],
        "router": [(2, 8), (4, 4), (4, 8), (8, 2), (8, 4), (8, 8)],
    }
    for layer, cfgs in plan.items():
        w = state[layer]
        n = 1024 if layer == "unpooler" else w.shape[0]
        k = w.shape[1]
        x = torch.randn((1, k), device=dev).to(torch.bfloat16)
        ref = x.double() @ w[:n].double().T
        odt = torch.float32 if layer == "router" else torch.bfloat16
        data = {"x": x, "config": {"m": 1, "n": n, "k": k, "layer": layer}}
        t_v5 = bench(lambda: kernel_fn(data, state))
        t_cb = bench(lambda: x @ w[:n].T) if layer != "router" else float("nan")
        res = []
        for wpr, u in cfgs:
            out = torch.empty((1, n), device=dev, dtype=odt)
            run = lambda: mod.gemv1(w, x, out, n, wpr, u)
            run()
            torch.cuda.synchronize()
            err = float((out.double() - ref).norm() / ref.norm())
            res.append((bench(run), f"w{wpr}u{u} e{err:.0e}"))
        res.sort()
        lines.append(f"{layer} m1 v5 {t_v5:.1f} cublas {t_cb:.1f} | " + " ; ".join(f"{t:.1f}/{s}" for t, s in res))
        print(lines[-1], flush=True)
    lines.append(f"total {_time.time() - t0:.0f}s")
    raise RuntimeError("PROBE\n" + "\n".join(lines))
