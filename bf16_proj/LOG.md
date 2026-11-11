# muse_bf16_proj experiment log

Env: judge H200 (sm 1980 MHz, driver 580.178.04), image judge@sha256:9a7f287b…; local has no GPU
(correctness via TRITON_INTERPRET=1, triton 3.4.0, with a test-only bf16 tl.dot fix).

| # | job | kind | change / hypothesis | result |
|---|---|---|---|---|
| v1 | e148dd7fe9644404 | check | split-K Triton (BN64, pow2 split to >=256 units), router hi/lo split in-register | pass, **0.834x** [0.822, 0.847]; router m<=32 1.4-3.3x, but router m128/256 0.26x/0.16x (register spill), qkv 0.61-0.78x (288 units > 264 slots: tail wave) |
| t1 | (tune.py) | check | sweep BN/BK/warps/stages/split in prepare, CUDA-graph replays, cold L2 | BN32 BK128, balanced non-pow2 split (qkv s7); router hi/lo pre-split fixes m128/256 |
| v2 | 2c08e31f35e64c46 | check | balanced split-K, tuned per layer, router pre-split hi/lo, cuBLAS for bf16 m>32 and unpooler m>8 | pass, **1.198x** [1.180, 1.216]; per load: d1 1.18, d16 1.19, d32 1.17, v1 1.52, v16 1.10, v32 1.08 |
| t2 | (tune.py) | check | judge env torch 2.14.1+cu130, triton 3.8.0; tile-major repack, m128/256 Triton, read-flush | tiled ~1-2% on qkv/pooler; Triton loses at bf16 m>=128 except pooler m128 (+6% in probe) |
| v3a | 39ef4173b0ce4b15 | check | tiled qkv/pooler/unpooler-m16, pooler m128 Triton (BM128, 8 warps), router split tweaks | **FAIL**: pooler m128 recheck rep 21, rel err 1.2e-2 |
| s1 | (stress.py) | check | 1500 cold-L2 graph replays per split-K config x 3 reduction protocols | P0 race: BM128/8-warp 95/1500 bad (reducer re-zeroed ws before sibling threads read it); barrier (P1) or atomic xchg (P2): 0 bad everywhere |
| v3b | 159408fa21424ac7 | check | + barrier between reducer's read and re-zero | pass, 1.197x [1.179, 1.215]; pooler m128 0.98x (probe gain didn't carry) |
| v3 | 9e47ae37166f4e48 | check | pooler m128 back to cuBLAS | pass, **1.211x** [1.193, 1.229] |
| v3 | 341a55f920614b8a | **scored** | same file as v3 check | pass, **1.212x** [1.194, 1.230]; rank 542 (leader 1.637x, top 20 >= 1.534x), provisional |
| t3 | (tune3.py) | check | swap-AB TMA kernel (W as wide MMA operand, all of m in one tile) vs current vs cuBLAS | decode: 1-5% better than current on every BF16 shape; m128: only pooler +5%; m256: clearly worse than cuBLAS |
| p4 | bb56e632625543f3 | check | roofline: pure streaming read per weight size; 12-bit weight format read + decode cost | pure read 4.0-4.4 TB/s: ours within ~1.8-2us (pool/qkv) of it; 12-bit read saves 3.8us on pooler, but plain-Triton decode is ALU-bound (22.1us raw vs 20.5 for plain bf16 read): not worth it without PTX byte-permute decode. Top-15 high bytes cover 99.966% of weights |
| v4 | 5b7e201c9f8d4dc0 | check | swap-AB TMA for BF16 m<=32 (o, unpooler m16/32 now Triton too); router kernel kept; tile-major copies dropped | pass, **1.231x** [1.213, 1.250]; o 1.12-1.14x, unpooler m16/32 1.06x, pooler 1.14-1.16x |
| v4 | 5dee8afa497445ec | **scored** | same file as v4 check | pass, **1.227x** [1.208, 1.245]; rank 522 (was 542), provisional |
| t5 | d84ff26489b9433d | check | m=128/256 BF16: normal (wide-N wgmma) + swap-AB TMA, with/without warp specialization, vs torch + SGLang dispatch | SGLang dispatch == torch cuBLAS exactly; only pooler m128 beats it (swap-AB 64,128,4w,4st,s2: 26.9 vs 29.3 raw); warp specialization no gain; m256 Triton 9-35% slower |
| v5 | fb5535f6d4e74e70 | check | v4 + pooler 65-128 tokens on swap-AB | pass, 1.222x [1.204, 1.240] (v4: 1.231/1.227: same within noise); pooler m128 1.04x |
| p6 | 49012471d48a4e92 | check | split-K epilogue cost: full hand-off vs atomics-only vs plain stores vs no write | atomics free; the counter + re-read + store hand-off costs ~1.1-1.4us per split-K shape; without it every shape is within 0.6-1us of a pure streaming read |
| p7 | 4a25d6aa726d4af2 | check | hand-off replacements: reduce kernel chained by PDL, plain reduce kernel, router zeroed-output + direct atomics | PDL -0.2..0.4us at best and inconsistent (qkv m16 +0.9us); others no gain: a second node costs about what the hand-off does |
| p8 | ab8a4fe21cf6446b | check | no split-K at all: BN 16/32 row tiles over full K, TMA 4-10 stages, +/- warp specialization | loses everywhere (slice 13.4 vs 9.5, qkv 21.4 vs 16.0) except pooler m16/32 (= current): one CTA per SM can't stream its share |

Known-good: submission_v5_good.py (v4 + pooler m128; equal to v4 within noise). Scored best: v4 (1.231x check). Previous: submission_v3_good.py (1.211x check, race-fixed). submission_v2_good.py lacks the barrier fix: don't use it.
Remaining: bf16 m=128/256 at parity (cuBLAS); unpooler m16/32 at parity; pooler ~73% of HBM BW.

## Session 2026-10-27 (afternoon UTC): native CUDA path

Baseline preserved: submission_v5_good.py sha256 fe605a68...4adff (== submission.py). Judge env: torch 2.14.1+cu130,
triton 3.8.0, CUDA 13.0 nvcc at /usr/local/cuda, okc 0.7.2. Local: no GPU; nvcc 12.8 (conda, scratch) for compile checks only.

Rules re-read (llms.txt "Kernels"): native CUDA is the documented path ("compile it inside the file with PyTorch's
load_inline"); DeepGEMM, FlashInfer, TileLang, CuTe DSL, CUTLASS are in the image. My earlier "subprocess check blocks
nvcc" assumption was wrong.

Score model verified: recomputing the geomean from the v4 scored run's per-shape data gives 1.2265 (= official).
Per us saved: decode 1 +0.35%, verify 1 +0.34%, decode 16 +0.27%, decode 32 +0.26%, verify 16 +0.22%, verify 32 +0.18%.

| # | job | kind | change / hypothesis | result |
|---|---|---|---|---|
| p9 | a2137503808244ce | check | load_inline feasibility; library inventory | load_inline OK, 21 s compile, no static-check flags; deep_gemm 0.2.1 (bf16_gemm_nt), flashinfer 0.7.0 (mm_bf16), cutlass 4.8.0 |
| p10 | 82cd8989c8c74132 | check | E1: DeepGEMM / cuBLASLt / FlashInfer(cuDNN) vs cuBLAS at m=128/256 | all within +-0.3us of cuBLAS on 7/8 shapes; pooler m256: deep_gemm 33.50 / cudnn 33.57 vs 35.30 raw (~6% on one shape, ~+0.3% aggregate: inside noise, not adopted) |
| p11 | 047461083fa14101 | check | E2 v1 CUDA skinny kernel | build failed: -std=c++17 vs torch 2.14 headers (C++20). Fixed by -std=c++20 |
| p11b | a5e5ae846f9249ef | check | E2 v1: lane-packed W (A fragments, 16 B/lane), mma.m16n8k16, 16-row CTAs, warp + cluster/DSMEM reduction, x B-fragments from global | correct everywhere, slower everywhere (qkv m32 65.7 vs 16.9; slice m1 9.98 vs 9.38): 16-row CTAs re-read x ~2x the weight bytes from L2, serially ahead of each MMA |
| p12 | dc69b4e5316246da | check | E2 v2: 64-row CTAs, x chunk staged once per CTA in padded smem, 16 loads in flight per warp per chunk | still slower (o m32 47.8 vs 10.1): epilogue smem atomics had an 8-way bank conflict (row stride 8*NT floats) |
| p13 | 48eb7a3a20014d21 | check | E2 v2 + conflict-free lane-major epilogue | o m32 47.8 -> 18.3, but still slower on every shape (slice m1 11.3 vs 9.3, pool m1 24.9 vs 22.2, qkv m1 18.1 vs 15.9). Remaining cause (hypothesis, not measured): no cross-chunk pipelining -- each chunk is load -> sync -> stage x -> sync -> MMA with 1-2 CTAs/SM, so HBM latency isn't hidden the way Triton's multi-stage TMA pipeline hides it |

E2 status: correct (FP64-checked on 18 shapes incl. m=5, 27, router hi/lo, split 1-16, plus exact CPU emulation of the
fragment indexing) but not competitive. Its premise (a cluster/DSMEM reduction instead of the ~1.2us global hand-off)
is still untested in isolation, because the main loop doesn't yet stream as well as v5's. Prototype kept in
skinny_cuda.cu. Next step if resumed: multi-stage cp.async.bulk / mbarrier pipeline for W and x, then re-measure
against v5 on slice/qkv/pool m=1 before touching anything else.

No scored submission this session: no candidate beat v5 on aggregate.

| v5 | a8c6bf3579e44f7f | **scored** | submission_v5_good.py (sha256 fe605a68...4adff) | pass, 1.224x [1.205, 1.242]; same as v4 within noise; board keeps v4 (1.2265), rank 557 |
| p14 | 641f1687a6684a39 | check | E2 v3: 4-stage cp.async ring (16 KB W + x tile per stage, ~72 KB in flight per CTA, warp w -> step w), lane-major epilogue, cluster/DSMEM reduce; skinny_cuda.cu sha256 in session notes | correct everywhere; m=32 improved over v2 (qkv 29.2 vs 45.3, pool 34.0 vs 45.4, o 16.6 vs 18.3) but still slower than v5 on every shape: slice m1 11.9 vs 9.6, router m1 11.1 vs 9.7, qkv m1 19.3 vs 16.1, pool m1 27.0 vs 22.4, unpooler m16 26.1 vs 21.2 |

E2 stopped after 3 iterations (v1-v3). Each fixed a diagnosed flaw (x reuse, bank-conflicted epilogue, no pipelining) and the
gap to v5 stayed 15-45% at m=1. With no local GPU or profiler, each further hypothesis costs a judge run and can only be
tested blind; the upside if it eventually matched v5's streaming is the ~1.2us hand-off per split shape. Not pursued
further; v5 remains the candidate. Prototypes: skinny_cuda_v2.cu (v2), skinny_cuda.cu (v3), probe14.py (harness).

## Session 2026-10-30: lossless 12-bit weight compression (submission_zip.py, probe17/18)

Idea: TASK.md allows lossless compression of BF16 weights. These are Gaussian: exponent entropy 2.54 bits, and the
high byte (sign + exponent bits 7..1) fits an 8-value window for all but ~1.2e-4 of weights. Format: low byte raw +
4-bit nibble (sign, window index) = 12 bits/weight (25% fewer bytes); out-of-window weights are coded as the window's
smallest value and corrected exactly by a sparse per-(64-row tile, 128-col block) list (x * delta in FP32, or two
entries when delta isn't FP32-exact). Encoding verified exact (CPU), kernel bit-identical to v5's in the interpreter
(interp_bf16_fix.py patches the interpreter's BF16 dot), PTX decode (prmt, 8 ops / 4 weights) emulated bit-exact.

| # | job | change | result |
|---|---|---|---|
| z1 | 4fafbdabed0f45d3 | zip kernel, nibble pairs interleaved (join+reshape), plain-Triton decode, CSR exceptions | pass, 0.809x: every zip shape 1.4-1.9x slower |
| z2 | 156cf8cf368c4cd7 | split-half nibble layout (no join), inline-PTX decode, pooler m128 back on plain | pass, 0.909x: still 1.2-2x slower |
| p17 | 7fd3e089b1e7422d | sweep BK/stages/warps/split per layer | best zip config slower than v5 everywhere (o 12.9 vs 7.6, qkv 20.9 vs 13.0, pooler 26.5 vs 21.6, unpooler 26.3 vs 21.2) |
| p18 | 24799091dcc8464b | exceptions: CSR vs none vs dense fixed-capacity table | even with NO exceptions zip loses: o 9.8 vs 7.5, qkv 15.1 vs 13.2, pooler 22.9 vs 21.2, unpooler 22.8 vs 21.0; CSR costs +3-5 us (dependent cold-L2 loads), dense +1-2 us |

Why (sm90 TTGIR from an offline compile): the decoded A operand goes smem -> registers in the bf16 dot-operand
layout (kWidth 2: 2-byte ld.shared per thread), then each warp_group_dot is followed by a wait(0), so every step is a
serial TMA-wait -> lds -> decode -> wgmma -> wait chain. v5's smem-operand wgmma pipelines; this doesn't. Saving 25%
of the bytes would need the decode off the critical path (warp-specialized producer decoding into smem for async
wgmma), which Triton doesn't give here and the earlier native-CUDA attempts (E2) couldn't stream as well as v5.
Stopped; submission.py stays v5. submission_zip.py kept as the reference implementation of the format.

## Session 2026-10-31: native CUDA m=1 GEMV, mma.sync skinny kernel

Leaderboard: leader 1.652x, top 20 >= 1.59x, us 1.2265 (rank 860/1540). Gap analysis from the v4 scored run: small
shapes lose a fixed 2-3.5 us each to pure streaming (unpooler slice 5.3 us net for 8.4 MB, ~2 us of reading), and
verify 16/32 (m=128/256) sit at 1.0x: two of six loads with no gain. Top entries must be near the read roofline on
every shape. TileLang 0.1.12 and CUTLASS C++ headers (flashinfer 4.5.0, deep_gemm 4.2.1) are in the image.

| # | job | kind | change / hypothesis | result |
|---|---|---|---|---|
| p19 | fb8caa92633d4348 | check | inventory + TileLang swap-AB split-K (atomics) at m=16/128/256 | TileLang compiles in ~3 s, correct; no faster than Triton/cuBLAS (pool m128 30.2 vs cuBLAS 29.0 raw) |
| p20 | 6bd5c61e7c0446de | check | gemv1.cu: m=1 GEMV, warp per row segment, smem x (cp.async), register double-buffered 16 B streaming loads, smem reduce | beats v5 on all m=1 shapes (raw): pool 21.5 vs 22.5, qkv 14.8 vs 15.8, o 10.1 vs 10.3, slice 8.1 vs 9.4, router (fp32 W) 8.1 vs 9.6 |
| p21 | b147a34fa0d549f0 | check | skinny_mma.cu v1: mma.sync, W repacked in A-fragment order, 4 B x fragments from global, warps over rows, cluster/DSMEM reduce | correct (odd m too), 1.2-3x slower than v5: ~4 warps/SM in flight, x loads scale with m |
| p22 | 22a831ecc4594c58 | check | v2: K-permuted repack (16 B x fragments), warps split K + smem atomics, R row tiles/warp | m=8 slice = v5 (9.4 vs 9.5); m=16 1.2-1.4x slower, m=32 2-2.5x slower (x re-read per CTA from L2) |
| v6 | 636b442da72e492c | check | v5 + gemv1 for every m=1 shape | pass, **1.265x** [1.246, 1.284]; decode 1: qkv 1.25, o 1.15, pool 1.21, unpool 1.72, router 2.52 |

Known-good: submission_v6_good.py. Next levers: m=128/256 (verify 16/32 at 1.0x), split-K hand-off at m=8..32.
| v6 | cb4769f112e042f1 | **scored** | submission_v6_good.py (sha256 a4250df1601fbc85...) | pass, 1.259x [1.240, 1.278]; rank 721 (was 860) |
