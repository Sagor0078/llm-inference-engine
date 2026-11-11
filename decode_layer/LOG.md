# muse_decode_layer experiment log

| # | job | change | result |
|---|---|---|---|
| v1 | 74705e26cb09430a | SGLang's layer + muse_bf16_proj v5 kernels (qkv_gate, o, pooler, unpooler) + muse_attention_decode v7 kernels (prologue, attention, combine); router, routing, Marlin experts, boundaries unchanged; dead 512 MB RoPE table removed | pass, **1.089x** [1.073, 1.106]; decode 1.09-1.17x, verify 1 1.15x, verify 16/32 1.007x |

Profile (v1): verify 32 sliding: Marlin MoE 511 of 699 us (73%); decode 1: ~29 kernels, many 4-6 us torch elementwise /
reduce / layer_norm / gemv ones around the router, pre-expert norm, expert-id remap and unpooler pad.
| v2 | 22a13bf080524cd7 | + post-expert norm and weighted sum as one Triton kernel (SGLang's falls back to ~14 PyTorch kernels up to 16 tokens; see muse_moe_w4a16 probe m2) | pass, **1.172x** [1.155, 1.190]; decode 1.18-1.33x, verify 1 1.34x, verify 16/32 ~1.0x |
| v3 | bdb55c3506e64601 | + router as one fused hi/lo Triton kernel at every T (FP32 input up to 8 tokens: x split into BF16 hi + lo, 3 dots, ~4e-6 rel err vs FP64), own split-K workspace (it runs beside the pooler on the side stream) | pass, **1.197x** [1.179, 1.215]; verify 1 1.34 -> 1.40x, decode 16 1.23 -> 1.30x |
| v4 | a3da508976924273 | + one kernel for the pre-expert RMSNorm x (1 + w) and the slot -> local expert map (replaces id shift, modulo, expert-map index and F.rms_norm) | pass, **1.254x** [1.235, 1.273]; decode 1 1.45x, verify 1 1.50x, decode 16 1.36x |
| v5 | 744ab6e0f1394749 | + both norm/residual boundaries as Triton kernels (FP32 math, alpha/beta precomputed); the post-FFN one reads the 1024-column unpooler slice directly, so no F.pad kernel | pass, **1.257x** [1.238, 1.276]: = v4 within noise (verify 1 1.50 -> 1.53x); SGLang's boundaries were already fused |
| v6 | ef88c426afd748cf | + routing (sigmoid, biased top-8 by 8 argmax rounds, renorm) folded into the pre-expert kernel | 1.250x [1.232, 1.269]: no gain: one launch fewer, but the routing left the side stream (where it overlapped the pooler) for the critical path. Kept as submission_v6_routing.py |
| v7 | c29f19fe6ff94033 | v5 + Marlin block layout from one Triton kernel (replaces moe_align_block_size) and 16 rows per Marlin block at 9-32 tokens (muse_moe_w4a16 probe_m3) | pass, **1.271x** [1.252, 1.290]; decode 16 1.36 -> 1.44x, verify 1 1.55x |

Known-good: submission_v7_good.py (sha256 25ea1272b471ac6a...). Earlier: submission_v5_good.py (sha256 b42b34e92fec6a3f...). Earlier: submission_v4_good.py (sha256 4a476ea618dc67a8...). Earlier: submission_v2_good.py (sha256 bcd35fe49f08f7cf...). Previous: submission_v1_good.py (sha256 91235164ab76b1f6...).
