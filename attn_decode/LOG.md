# muse_attention_decode experiment log

Judge env: torch 2.14.1+cu130, triton 3.8.0, H200. Local: no GPU; correctness via TRITON_INTERPRET=1 with the judge's
reference.py checks on CPU (12 reduced cases incl. page 64, dp heads, kv_scale 1/64, edges, coherent, q_len 8), plus two
test-only interpreter fixes (bf16 tl.dot; FP32->FP8/BF16 RTNE without exponent carry).

SGLang baseline (job abfb130433bd4929, baseline.py as a submission): 0.995x. Sliding calls 20-35us (floor 0.1-3.7):
5-kernel chain overhead. Full layers at ~33% of HBM floor (FA3, page size 1 gather).

| # | job | change | result |
|---|---|---|---|
| v1 | 69293f8fecbc4fb1 | 3 Triton kernels: prologue (norm/RoPE/cache write/q->E4M3), split-KV attention (per KV head), combine+gated norm | pass, **1.210x** [1.192, 1.231]; sliding 1.6-2.2x; verify full 0.84-0.85x |
| v2 | 3ccd846f686343d6 | head pairs for decode (128 B rows, block-diagonal q), FP8 P@V | 1.120x: full decode +10-15%, but FP8 P@V slower on verify (Triton operand conversion), B1 full regressed (626 splits) |
| v3 | 75602d52613748d0 | unmasked fast path via runtime if inside the key loop | **0.837x**: a branch in the loop stops Triton pipelining the loads. Reverted |
| v3b | eecaa87daaa443de | BF16 P@V for verify, chunk >= 512 | 1.100x: fixes B1 full, but only 4 programs on B1 sliding |
| a1 | (probe_a1.py) | sweep BLOCK_N / warps / stages / P@V dtype / pairing per shape | full decode B32 248us (bn128 st3 bf16 pair) vs SGLang 388; verify full bn64 st2 412 vs 394 |
| v4 | c0cfd8c22e5a48ba | tuned per-shape configs, split cap 256 | pass, **1.390x** [1.369, 1.411] |
| v5 | f46e5dcde71b4e79 | page indices prefetched one block ahead | 1.364x: decode full -5%, verify full +10% |
| a2 | (probe_a2.py) | loads-only vs full kernel, B32 full | **gather alone = 199us decode / 248us verify (floor ~131)**: Triton's page-size-1 gather runs ~65% of peak; math adds 50 / 160us |
| v6 | 64b8632b377d40d0 | tl.multiple_of(row offsets) for 16 B vector loads | 1.386x: no change (already vectorized). Rejected |
| a3 | (probe_a3.py) | whole-call sweep: program target, split cap, prefetch | decode prefetch +5-7%; sliding decode B32 target 528; batched verify split cap 32 (sliding verify B16 29.5 -> 24.6) |
| v7 | 7c91c5cbccb34eaa | a3's rules (prefetch decode only) | pass, **1.412x** [1.391, 1.440] |
| v7 | 91e0b24e45394b84 | **scored**, same file (submission_v7_good.py) | pass, **1.434x** [1.413, 1.461]; rank 622 (leader 3.03x) |

Known-good: submission_v7_good.py (sha256 268f98279cce5b17...).
Remaining: verify full 0.95x (compute-bound: 64 query rows per KV head); full-layer gather ~65% of peak (leaders at ~3x
are at the floor); combine + prologue = 2 extra launches on small shapes.
