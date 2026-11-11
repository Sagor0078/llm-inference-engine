# Best submissions

The best version of each kernel, one file per challenge problem. Each file is self-contained
(native CUDA sources are embedded and compiled with `load_inline`), and its first line names the
problem, so it can be submitted directly:

```bash
okc check best_submissions/muse_bf16_proj.py --json
```

| File | Problem | Version | Speedup vs SGLang | Full history |
|---|---|---|---|---|
| [muse_bf16_proj.py](muse_bf16_proj.py) | Dense BF16 projections + FP32 router | v6 | 1.259x scored (1.265x check) | [bf16_proj/LOG.md](../bf16_proj/LOG.md) |
| [muse_attention_decode.py](muse_attention_decode.py) | Gated attention decode over paged FP8 cache | v7 | 1.434x scored | [attn_decode/LOG.md](../attn_decode/LOG.md) |
| [muse_moe_w4a16.py](muse_moe_w4a16.py) | Routed experts, NVFP4 weights | v2 | 1.174x scored (1.179x check) | [moe_w4a16/LOG.md](../moe_w4a16/LOG.md) |
| [muse_decode_layer.py](muse_decode_layer.py) | Whole decoder layer, decode/verify | v7 | 1.271x check | [decode_layer/LOG.md](../decode_layer/LOG.md) |
| [muse_prefill_layer.py](muse_prefill_layer.py) | Whole decoder layer, prefill | v3 | 1.133x submitted | [prefill_layer/LOG.md](../prefill_layer/LOG.md) |

These are the only submission files in the repository; the working `submission.py` in each
problem folder is kept local.
