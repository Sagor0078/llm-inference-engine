# muse_moe_w4a16 experiment log

| # | job | change | result |
|---|---|---|---|
| v0 | 14a4fd3c166d49d9 | baseline.py as a submission (SGLang Marlin path) | 1.002x. SGLang per call: bs1 77.0, bs16 188.3, bs32 206.2, bs8 150.4, bs128 458.3, bs256 622.1 us (floors 11-109 us). bs1 profile: Marlin 41.5 of 76.3 us, the rest ~15 small torch/glue kernels; bs256: Marlin 552 of 575 us |
| m1 | 71cc8a22d78e4cca | probe: Triton grouped GEMMs (align, GEMM1+SiLU, GEMM2, finalize; in-register NVFP4 dequant, even/odd nibble planes, exact BF16 weights x FP32 global scale) vs Marlin, tile sweep | correct (CPU: 6 cases, 0.28% vs 1.5% bound; GPU: < 2% vs Marlin), but **2.7-3x slower than Marlin at every bs** (bs256 977 vs 364, bs1 239 vs 85 us). Rejected: in-register dequant (~8 ALU ops/value) and strided activation loads; Marlin's bit-trick dequant + layout is far ahead |
| m2 | e7538baaef944931 | probe: Marlin path stage by stage, graph replays + kernel lists | SGLang's fused_post_expert_norm_weighted_sum falls back to 14 PyTorch kernels up to bs 16: 34.2 / 40.5 / 43.5 us at bs 1 / 8 / 16 (one 9 us Triton kernel at bs 32). Marlin GEMMs 54-111 us; map 7, align 6-9, SiLU-mul 6 us |
| v1 | 33ae4a6870f540cb | Marlin path + my single-kernel finalize (moe_triton._finalize_kernel; (1 + w) in FP32, off-GPU slots skipped by id) | pass, **1.143x** [1.125, 1.160]; bs1 1.47x, bs8 1.26x, bs16 1.21x, bs >= 32 1.0x |
| m3 | 6e9e18af56784985 | probe: Marlin block_size_m x workspace size per bs; dump of moe_align_block_size's format | SGLang's heuristic is best except bs 16: block 16 = 78.4 vs 92.0 us (block 8). Workspace size: no effect. Format: per-expert ascending slots padded with numel, off-GPU slots in a leading expert -1 block, npad = rows |
| v2 | fba916695f4d4c05 | v1 + one Triton align kernel in Marlin's format (also writes local ids; off-GPU slots omitted) replacing the expert-map index, its copy and moe_align_block_size; block 16 at 9-32 tokens | pass, **1.179x** [1.162, 1.197]; bs1 1.57x, bs8 1.33x, bs16 1.23x, bs32 1.04x, bs128/256 ~1.0x |
| m4 | 0e88e9dbbe904a51 | probe: library inventory for BF16 x NVFP4 (W4A16) kernels; Marlin's full signature | candidates: flashinfer.mm_bf16_fp4 (+ prepare_bf16_fp4_weights, group-16 E4M3 scales), fused_moe B12xW4A16 / CuTileNvfp4 runners, CuteDslBf16MoEWrapper (SM90 but BF16 weights only), deep_gemm.bf16_mega_moe (BF16). Marlin exposes no tile / SM-count knobs |
| m5 | ba66ab0d54e145c1 | probe: mm_bf16_fp4 on sm90 with swizzled scales (block_scale_interleave), both backends | **not available on H200**: cudnn "does not support capability 90"; cute-dsl "only SM100/103 and SM12x"; B12xW4A16 is SM120/121. No library W4A16 path on sm90 |
| v2 | 976becf695694e2c | **scored**, same file as the v2 check | pass, **1.174x** [1.156, 1.192]; rank 1030 (leader 2.69x), provisional |

### Option C: Triton grouped GEMM with a Marlin-style inline-PTX dequant

| # | job | change | result (raw us, incl. ~5 us graph cost) |
|---|---|---|---|
| m6 | 255fa547145f44c5 | moe_triton2: E2M1 -> BF16 via prmt + mask/shift (value x 2^-126, 0.5 as a BF16 subnormal) + fma.rn.bf16x2 x 2^126 + x E4M3 scale (exact; NumPy emulation 0 mismatches / 16,384; ptxas-compiled locally for sm_90; constexpr asm strings) | correct on all shapes; 1.6-2.4x faster than the arithmetic dequant (bs256 1457 -> 620, bs1 245 -> 140), still behind Marlin |
| m7 | e781a9f150c5499b | moe_triton3: + lossless nibble repack (low nibbles = first half of each BK chunk, high = second: contiguous activation halves) + scales loaded once per 16 values | bs1 119, bs8 125, bs16 127, bs32 172, bs128 406, bs256 539 |
| m8 | 67445779b3024112 | per-stage profile | GEMM1 at bs1 70 us = 0.87 TB/s (Marlin ~2.2): parallelism-starved (3 blocks x 64 tiles, full K serial per program); bs256 GEMM1 ~125 TFLOPS |
| m9 | d315e85f766940e9 | moe_triton4: + split-K for both GEMMs (zeroed FP32 workspaces, last-arriver SiLU-mul / store, barrier fix; workspaces verified clean after replays) vs the **v2 submission path** | still 1.5-1.65x slower than v2 at every bs: bs1 87.8 vs 56.8, bs16 109 vs 73, bs32 181 vs 114, bs128 385 vs 239, bs256 558 vs 371. All correct |
| m10 | 07cb72c1de194b57 | moe_triton5: + tile-major weights and scales ([E, N/BN, K/BK, BN, BK/2], lossless; bit-identical to row-major locally), H/I constexpr | tile-major beats row-major at every bs; vs v2: bs1 66.9/57.5, bs8 86.0/68.7, bs16 88.2/73.6, bs32 126.5/114.4, bs128 299.0/238.2, bs256 409.7/373.1 (1.10-1.26x behind) |
| m11 | 54722cae3e1f467d | wider tile sweep: BN 64-256, BK 128/256, BM 16-128, warps, stages, split | no better config: plateau 1.10-1.25x behind v2 (BM 128 out of shared memory) |

Note (m10 investigation): offline sm_90 compiles first showed byte loads and no cp.async -- an artifact of compiling
without the launcher's 16-byte pointer-alignment attributes; with them the kernels are vectorized and cp.async-pipelined.

Option C status: the dequant is no longer the gap (PTX path ~3 ops/value, exact), and split-K fixed decode parallelism,
but Marlin's GEMMs stay ~1.5x ahead everywhere. Kept: moe_triton2/3/4.py (with the exact PTX dequant), emulate_fp4 /
compile tests in the session scratchpad. Not adopted. (m9 also cost one failed run d-precursor 77b154f44c174764:
probe constants stripped by mistake.)

Known-good: submission_v2_good.py (sha256 ba60973e441def30...). Earlier: submission_v1_good.py (sha256 ca4dbbfc40cfe118...). Prototype kept: moe_triton.py (Triton GEMMs, rejected).
Next if resumed: keep Marlin, replace the small-bs glue (bs1: ~35 us across ~15 kernels) with one fused align kernel in
Marlin's sorted_ids/block format, and fuse SiLU-mul + finalize.

### Option D: hand-written / CUTLASS Hopper (wgmma) W4A16 GEMM

| # | job | finding |
|---|---|---|
| d1 | 3e822c2fc2964b7e | Image has CUTLASS 4.8.0 C++ headers (/opt/cutlass/include) + nvcc, incl. sm90 mixed-input wgmma collectives (plain and ptr-array / grouped, "rs": narrow operand converted in registers). |
| local | (CUTLASS v4.8.0 cloned to the scratchpad) | **Not usable for NVFP4 as is**: ConvertAndScale requires chunk_size == K or chunk_size % TileK == 0 (sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp can_implement), and NVFP4's scale group is 16, so TileK would have to be 16 (one wgmma K-step per mainloop iteration). Its E2M1 lookup-table path is for 8-bit (FP8) compute, i.e. W4A8, not BF16 activations. Block scales can't move outside the MMA (they vary along K within a tile). |

Remaining D route: a from-scratch sm90 kernel (TMA + mbarriers, packed weights -> registers, PTX dequant incl. per-16
scales, wgmma RS with activations from shared memory, split-K / grouped scheduling): ~400+ lines of CUDA/PTX, no local
GPU, ~8-20 judge runs/day. Not started.
| d2 | e14f776dbfeb4fc7 | moe_cuda.cu stage 1: grouped mma.m16n8k16 (swap AB), weights lane-packed (w4a16_pack.lane_pack, lossless; emulation 0 / 6,144 mismatches) straight to registers, exact PTX dequant, activations staged in padded smem, split-K hand-off; GEMM1 gated (silu(g)*u epilogue). nvcc sm_90a: 55-152 regs, no spills | **correct on first GPU run** (<= 0.6% vs v2 at every bs, stable across replays) but slower: bs1 78.6 / 56.5 (v2), bs16 116.2 / 72.3, bs32 187.6 / 114.2, bs128 393.2 / 239.6, bs256 626.9 / 371.6 -- behind the Triton tile-major path too |
| d3 | 42cff3a9569949ac | + register double-buffering of the weight loads | **worse**: bs1 88.2, bs16 127.2, bs256 854.0 (more registers -> lower occupancy; no latency gained) |

Option D stopped after d3: correct kernel, 1.4-2.3x slower than Marlin, and the first optimization made it slower.
The best non-Marlin path remains Triton tile-major (m10/m11, 1.10-1.26x behind v2). MoE submission stays v2.

