// Skinny BF16 projections (m <= 32): out[m, n] = x[m, k] @ W[n, k]^T, FP32 accumulation.
//
// W arrives pre-packed (lossless reorder, in prepare) as [n/16 stripes][k/16 steps][32 lanes][8 bf16]:
// lane l's 16 bytes are exactly its A fragment of mma.m16n8k16 for that (stripe, step), so a warp
// streams 512 contiguous bytes per fragment with one 128-bit load per lane and no staging.
//
// A CTA owns R stripes (16R rows) and one rank's slice of the k steps. It walks that slice in chunks
// of KC steps: every warp first issues the loads of all its A fragments in the chunk (R stripes x
// KC/WARPS steps, all independent), then the chunk of x is staged once in padded shared memory
// (tokens past m as zero), and the warps run their MMAs with x as the B operand (tokens as the MMA's n).
//
// Warp partials meet in shared memory (conflict-free atomics, C-fragment order); when a stripe group is split over SPLIT CTAs those CTAs form
// one thread-block cluster and rank 0 sums the others' partials through distributed shared memory:
// no global atomics, no arrival counter, no workspace that outlives a call.
//
// HI_LO (the router): W is the FP32 weight as a packed BF16 hi + lo pair, two MMAs per fragment into
// one FP32 accumulator; the output is FP32.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

constexpr int WARPS = 8;
constexpr int KC = 32;                 // k steps (of 16) per chunk
constexpr int SPW = KC / WARPS;        // steps per warp per chunk
constexpr int XS_STRIDE = KC * 16 + 8;  // bf16 per staged x row: +8 keeps B-fragment reads conflict-free

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

// NT n-tiles of 8 tokens (m <= 8 NT); R stripes per CTA.
template <int NT, bool HI_LO, int R>
__global__ void __launch_bounds__(WARPS * 32) skinny_kernel(const uint4* __restrict__ w, const uint4* __restrict__ w_lo,
                                                            const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
                                                            int m, int n, int k, int split) {
  __shared__ __align__(16) __nv_bfloat16 xs[8 * NT][XS_STRIDE];
  // This CTA's partial in C-fragment order, [stripe][n-tile][element][lane]: each atomic below
  // touches 32 consecutive words (no bank conflicts); rows and tokens are decoded on output.
  __shared__ float part[R][NT][4][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int stripe0 = blockIdx.x * R;
  const int rank = blockIdx.y;
  const int ksteps = k >> 4;
  const int s0 = (int)((long long)rank * ksteps / split);
  const int s1 = (int)((long long)(rank + 1) * ksteps / split);
  const int g = lane >> 2, t = lane & 3;

  for (int i = threadIdx.x; i < 16 * R * 8 * NT; i += WARPS * 32) (&part[0][0][0][0])[i] = 0.f;
  __syncthreads();  // zeroed before any warp adds (a rank may have no chunks)

  float acc[R][NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.f;

  for (int cs = s0; cs < s1; cs += KC) {
    const int ce = min(cs + KC, s1);
    // 1. This warp's A fragments for the chunk, all in flight at once.
    uint4 a[SPW][R];
    uint4 al[HI_LO ? SPW : 1][R];
#pragma unroll
    for (int i = 0; i < SPW; ++i) {
      const int st = cs + warp + i * WARPS;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (st < ce) {
          const size_t off = ((size_t)(stripe0 + r) * ksteps + st) * 32 + lane;
          a[i][r] = ld_stream(w + off);
          if (HI_LO) al[HI_LO ? i : 0][r] = ld_stream(w_lo + off);
        }
      }
    }
    // 2. Stage x[:, 16 cs .. 16 ce) once for the CTA (16-byte pieces; tokens past m as zero).
    __syncthreads();  // the previous chunk's reads of xs are done
    const int cols = (ce - cs) * 16;
    const int pieces = 8 * NT * (cols / 8);
    for (int p = threadIdx.x; p < pieces; p += WARPS * 32) {
      const int tok = p / (cols / 8), c8 = p % (cols / 8);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (tok < m) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)tok * k + (size_t)cs * 16) + c8);
      *reinterpret_cast<uint4*>(&xs[tok][c8 * 8]) = v;
    }
    __syncthreads();
    // 3. MMAs: B[k][tok] = xs[tok][16 (st - cs) + k].
#pragma unroll
    for (int i = 0; i < SPW; ++i) {
      const int st = cs + warp + i * WARPS;
      if (st < ce) {
        const int kc = (st - cs) * 16 + 2 * t;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[j * 8 + g][kc]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[j * 8 + g][kc + 8]);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            mma_bf16(acc[r][j], a[i][r], b0, b1);
            if (HI_LO) mma_bf16(acc[r][j], al[HI_LO ? i : 0][r], b0, b1);
          }
        }
      }
    }
  }

  // C fragment: acc[r][j][0..1] = (row g, tokens 8j + 2t, +1), [2..3] = (row g + 8, same tokens).
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        atomicAdd(&part[r][j][e][lane], acc[r][j][e]);

  const int row0 = stripe0 * 16;
  if (split > 1) {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();  // every rank's partial is complete in its shared memory
    if (rank == 0) {
      for (int i = threadIdx.x; i < 16 * R * 8 * NT; i += WARPS * 32) {
        float v = (&part[0][0][0][0])[i];
        for (int q = 1; q < split; ++q) v += cluster.map_shared_rank(&part[0][0][0][0], q)[i];
        (&part[0][0][0][0])[i] = v;
      }
    }
    cluster.sync();  // the others' shared memory stays alive until rank 0 has read it
    if (rank != 0) return;
  } else {
    __syncthreads();
  }
  for (int i = threadIdx.x; i < 16 * R * 8 * NT; i += WARPS * 32) {
    const int r = i / (NT * 128), j = (i / 128) % NT, e = (i / 32) & 3, l = i & 31;
    const int row = r * 16 + (l >> 2) + ((e & 2) ? 8 : 0);
    const int tok = j * 8 + (l & 3) * 2 + (e & 1);
    if (tok < m) {
      const float v = (&part[0][0][0][0])[i];
      if (HI_LO)
        reinterpret_cast<float*>(out)[(size_t)tok * n + row0 + row] = v;
      else
        reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row0 + row] = __float2bfloat16_rn(v);
    }
  }
}

template <int NT, bool HI_LO, int R>
static cudaError_t launch_t(const void* w, const void* w_lo, const void* x, void* out, int m, int n, int k, int split,
                            cudaStream_t stream) {
  auto kern = skinny_kernel<NT, HI_LO, R>;
  if (split > 8) {
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return e;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(n / (16 * R), split, 1);
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
  return cudaLaunchKernelEx(&cfg, kern, static_cast<const uint4*>(w), static_cast<const uint4*>(w_lo),
                            static_cast<const __nv_bfloat16*>(x), out, m, n, k, split);
}

// R = 4 stripes (64 rows) per CTA for BF16 layers, 2 for the router (twice the fragments per step).
// Returns a cudaError_t code (0 on success). n must be a multiple of 16 R, k of 16.
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
