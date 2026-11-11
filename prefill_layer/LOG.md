# muse_prefill_layer experiment log

Baseline and reference are identical to muse_decode_layer's; scored shapes are q_len 753 and 16384.

| # | job | change | result |
|---|---|---|---|
| v1 | 1d300c9c3dad4a0f | decode_layer v1 file with SGLang's FA3 attention restored for q_len > 8 (all prefill); projections fall back to cuBLAS at these sizes | pass, **1.000x** [0.985, 1.015]: parity, as expected |

Profile: Marlin MoE GEMMs 65-69% of every shape (turn 1 sliding: 808 of 1,173 us; first turn sliding: 11.5 of 17.6 ms,
~17% of BF16 tensor-core peak), cuBLAS projections ~18-23%, FA3 ~2%. The only large lever is the MoE GEMM at large M;
the Triton grouped GEMM (moe_w4a16/moe_triton.py) lost to Marlin by 2.7x, so no candidate beats parity yet.
Kept: submission_v1_parity.py (sha256 fa23951e5168332f...).

| p1 | 8d0344dcaf744bc4 | probe: the MoE experts at prefill sizes on this layer's weights: SGLang's Marlin path vs the tile-major Triton grouped GEMM (moe_triton5, PTX dequant) with large tiles (BM 64/128, BN 64/128, BK 64/128, warps, stages) | Triton slower and the gap grows with M: 753 tokens (1,545 local routes) 873 vs Marlin 674 us; 3,012 tokens (7,249 routes) 4,897 vs 2,721 us (Marlin ~27% of BF16 peak there, Triton ~15%); 16,384 not reached in the 250 s budget. No path to beat Marlin at prefill sizes with these kernels |

Status: parity (v1). The MoE GEMM is 65-69% of every prefill shape and neither the Triton (m-series) nor the CUDA (d-series,
muse_moe_w4a16) W4A16 kernels beat Marlin; the rest is cuBLAS projections (~20%) and FA3 (~2%). Not submitted.

| p2 | (probe) | first full grouped path in a probe: dequant + torch._grouped_mm + glue | rel err NaN: torch.index_add over uninitialized pad rows (0 x NaN). Fix: never read pad / past-offs rows |
| p3 | 48261b51dc41446d | torch._grouped_mm alone + exact vectorized dequant | dequant exact, all 32 experts 1,483.5 us; 3,012 tok (7,264 rows) GEMMs 993.7 us vs Marlin path 2,733.9; 16,384 tok (34,560 rows) 4,629.6 us (751 TFLOPS) vs Marlin 13,609.9 |
| v2 | 137db5ada58e4598 | experts >= 2,048 tokens: in-call exact NVFP4 -> BF16 dequant to scratch, atomic count + scatter (groups padded to 16), grouped_mm, per-expert-gs SiLU-mul, grouped finalize (gs2, post-norm, weighted sum) | pass, **1.079x** [1.063, 1.095]; turn 1 1.00/1.00, turn 4 0.90/0.91 (dequant 1.5 ms doesn't pay at 3K tokens), first turn 1.41/1.44/1.30/1.24. Kept submission_v2_good.py |
| v3 | 4aae8522ce704841 | v2 + threshold 8,192 (turn 4 back on Marlin), dequant on a side stream from the start of the call (overlaps attention/projections), contiguous-store dequant (BLOCK 4096), one-program-per-row SiLU | pass, **1.132x** [1.115, 1.149]; turn 1 1.00/1.00, turn 4 1.00/1.00, first turn 1.46/1.50/1.33/1.28. 16K profile: grouped GEMMs 4.43 ms, cuBLAS projections 4.2 ms (both ~peak), dequant 1.36 ms (barely overlapped: busy 11.56 vs sum 11.77), norm-residual 0.8, SiLU 0.23, FA3 0.31. Kept submission_v3_good.py |
| v4 | 7ee26e8c644e44d5 | v3 + high-priority dequant stream | 1.131x [1.114, 1.148]: no change (the GEMMs fill every SM). Reverted to v3 |

Status: v3 submitted: job 849f502a63f04a41, **1.133x** [1.116, 1.150].
