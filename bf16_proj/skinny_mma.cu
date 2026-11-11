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
