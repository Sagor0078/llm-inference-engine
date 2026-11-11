// Grouped W4A16 (NVFP4 weights x BF16 activations) GEMM for the routed experts, mma.m16n8k16 with the weights as the
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
// of K. Weight loads are register double-buffered: the next U steps are in flight while the current U are computed. The block's activations are staged chunk by chunk in padded shared memory (the B operand). With SPLIT > 1,
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

  // Weight / scale words of k steps [b, b + U), into registers (steps past s1 skipped).
  auto load_batch = [&](int b, uint32_t (&w)[U], uint32_t (&sc)[U], uint32_t (&wu)[GATED ? U : 1],
                        uint32_t (&su)[GATED ? U : 1]) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (b + u < s1) {
        w[u] = ld_stream_u32(wc + base_g + (size_t)(b + u) * 32);
        sc[u] = ld_stream_u16(wsc + base_g + (size_t)(b + u) * 32);
        if (GATED) {
          wu[GATED ? u : 0] = ld_stream_u32(wc + base_u + (size_t)(b + u) * 32);
          su[GATED ? u : 0] = ld_stream_u16(wsc + base_u + (size_t)(b + u) * 32);
        }
      }
    }
  };
  // Register double-buffering: batch b + U's loads are in flight while batch b is dequantized and multiplied.
  uint32_t cw[U], cs_[U], cwu[GATED ? U : 1], csu[GATED ? U : 1];
  uint32_t nw[U], ns_[U], nwu[GATED ? U : 1], nsu[GATED ? U : 1];
  if (s0 < s1) load_batch(s0, cw, cs_, cwu, csu);
  int cs = s0;  // first k step of the staged activation chunk
  for (int b = s0; b < s1; b += U) {
    if ((b - s0) % KC == 0) {
      // Stage the block's activations for k steps [b, min(b + KC, s1)): rows past the block's slots as zero.
      cs = b;
      const int ce = min(b + KC, s1);
      __syncthreads();  // the previous chunk's reads of xs are done
      const int pieces = BT * (ce - cs) * 2;  // 16-byte pieces (8 bf16)
      for (int p = threadIdx.x; p < pieces; p += WARPS * 32) {
        const int r = p / ((ce - cs) * 2), c8 = p % ((ce - cs) * 2);
        const int slot = slots[r];
        uint4 v = make_uint4(0, 0, 0, 0);
        if (slot >= 0) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)(slot / row_div) * k + (size_t)cs * 16) + c8);
        *reinterpret_cast<uint4*>(&xs[r][c8 * 8]) = v;
      }
      __syncthreads();
    }
    if (b + U < s1) load_batch(b + U, nw, ns_, nwu, nsu);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (b + u < s1) {
        uint32_t a[4];
        dequant(cw[u], scale2(cs_[u] & 0xFF), scale2(cs_[u] >> 8), a);
        uint32_t au[4];
        if (GATED) dequant(cwu[GATED ? u : 0], scale2(csu[GATED ? u : 0] & 0xFF), scale2(csu[GATED ? u : 0] >> 8), au);
        const int kc = (b + u - cs) * 16 + 2 * t;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[j * 8 + g][kc]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[j * 8 + g][kc + 8]);
          mma(acc[j], a, b0, b1);
          if (GATED) mma(accu[GATED ? j : 0], au, b0, b1);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      cw[u] = nw[u];
      cs_[u] = ns_[u];
      if (GATED) {
        cwu[GATED ? u : 0] = nwu[GATED ? u : 0];
        csu[GATED ? u : 0] = nsu[GATED ? u : 0];
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
