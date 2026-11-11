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
