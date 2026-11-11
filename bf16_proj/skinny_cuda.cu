// Skinny BF16 projections (m <= 32): out[m, n] = x[m, k] @ W[n, k]^T, FP32 accumulation.
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
