# llm-inference-engine

**Muse Spark Open Kernel Challenge: my solutions.**
GPU kernels that make parts of **Muse Spark Open** inference faster than stock **SGLang** on one
NVIDIA H200, written for the [Open Kernel Challenge](https://openkernelchallenge.dev/).
The final kernel for each problem is in [best_submissions/](best_submissions/); each problem folder
holds the probes and prototypes I used to explore it, and a `LOG.md` recording every experiment with
its judge job ID.

All results below come from the official judge (H200, CUDA-graph replays, cold L2, paired against
SGLang on the same inputs). Leaderboard positions are provisional until the organizers' final review.

## Results at a glance

| Problem | Folder | What it computes | Best result | Status |
|---|---|---|---|---|
| `muse_bf16_proj` | [bf16_proj/](bf16_proj/) | Dense BF16 projections + FP32 router (`x @ W.T`) | **1.265x** (check), **1.259x** scored | Scored, v6 |
| `muse_attention_decode` | [attn_decode/](attn_decode/) | Gated GQA decode/verify over a paged FP8 KV cache | **1.434x** scored | Scored, v7 |
| `muse_moe_w4a16` | [moe_w4a16/](moe_w4a16/) | Top-8 of 256 routed experts, NVFP4 weights, SwiGLU | **1.179x** (check), **1.174x** scored | Scored, v2 |
| `muse_decode_layer` | [decode_layer/](decode_layer/) | A whole decoder layer at decode/verify sizes | **1.271x** (check) | v7 |
| `muse_prefill_layer` | [prefill_layer/](prefill_layer/) | A whole decoder layer at prefill sizes | **1.133x** submitted | Submitted, v3 |

The speedup is the judge's score: the geometric mean over loads of SGLang's time divided by mine.
1.0x means parity with SGLang.

**Just want the kernels?** The best submission for each problem is collected in
[best_submissions/](best_submissions/), one self-contained file per problem. The math each one
computes is in [Mathematical formulation](#mathematical-formulation).

## The approach, in one paragraph

Almost every decode-time shape here is **memory-bound**: the job is to stream the weights (or the
KV cache) from HBM once, at close to full bandwidth, with as few kernel launches as possible. So the
work was: measure where SGLang loses time (small-kernel chains, under-filled SMs, PyTorch fallbacks),
replace those pieces with fused Triton or native CUDA kernels, tune them per shape with CUDA-graph
replays from a cold L2 (as the judge times them), and keep SGLang's own kernels wherever nothing
beat them (cuBLAS at large m, Marlin for the NVFP4 GEMMs). Every candidate went through
`okc check` before a scored run, and the log records the attempts that failed as well as the ones that worked.

---

## 1. Dense projections: `muse_bf16_proj` ([bf16_proj/](bf16_proj/))

`out = x @ W.T` for the qkv_gate, o, pooler, unpooler (BF16) and router (FP32) weights, at
m = 1 to 256 tokens.

**What the submission does** ([muse_bf16_proj.py](best_submissions/muse_bf16_proj.py)):
- **m = 1:** a native CUDA GEMV ([gemv1.cu](bf16_proj/gemv1.cu), compiled in `prepare()` with
  `load_inline`). Each warp streams one row segment with 16-byte loads, double-buffered in registers,
  and the warps reduce in shared memory. There is no split-K hand-off and no pipeline to fill.
- **m ≤ 32, BF16:** a split-K Triton kernel using TMA and a "swap-AB" layout (`out^T = W @ x^T`), so a
  64-row weight tile feeds the tensor cores against all of m at once. Work units are sized to fill
  the SMs in one wave; partial sums are reduced with FP32 atomics into a persistent workspace, and
  the last unit writes the output, so each call is a single launch.
- **Router:** the FP32 weights are split once into a BF16 hi + lo pair (the split SGLang uses), and
  both halves go through the tensor cores into one FP32 accumulator in a single pass over the weights.
- **m > 32:** cuBLAS, which nothing I tried beat there, except the pooler at 65-128 tokens.

**Results:** v1 0.83x → v2 1.20x (balanced split-K, tuned per layer) → v4 1.23x (swap-AB TMA)
→ **v6 1.265x** (CUDA GEMV for m = 1; router at m = 1 is 2.5x faster than SGLang).

**What I tried that didn't pay off** (all in [LOG.md](bf16_proj/LOG.md)):
- A race in the split-K reduction showed up in only 95 of 1,500 replays. I found it with
  [stress.py](bf16_proj/stress.py) and fixed it with a barrier.
- Native CUDA `mma.sync` skinny kernels ([skinny_cuda.cu](bf16_proj/skinny_cuda.cu),
  [skinny_mma.cu](bf16_proj/skinny_mma.cu)): correct, but 15-45% slower than the Triton TMA pipeline.
- Lossless 12-bit weight compression (the high byte of a Gaussian BF16 weight fits an 8-value
  window): bit-exact, but the decode step stalls Triton's wgmma pipeline, so it was slower even
  though it reads 25% fewer bytes.
- DeepGEMM, cuBLASLt, FlashInfer/cuDNN and TileLang at m = 128/256: all within noise of cuBLAS.

## 2. Attention decode: `muse_attention_decode` ([attn_decode/](attn_decode/))

One decode step or 8-token verify block of grouped-query attention over a paged FP8 cache, with
QK-norm, a 2,048-token sliding window with RoPE (or full attention), and a gated, normalized output.

**What the submission does** ([muse_attention_decode.py](best_submissions/muse_attention_decode.py)), as three Triton kernels
in place of SGLang's chain of five:
1. **Prologue:** RMSNorm of q and the new keys, RoPE, writing the new K/V to the cache as E4M3,
   and the query rounded to E4M3.
2. **Split-KV attention:** one program per (request, KV head or head pair, key chunk). FP8 × FP8 logits
   on the tensor cores, online softmax in FP32, BF16 P @ V. Masks are causal plus the sliding window,
   and lengths are read on the device, so the kernel stays CUDA-graph safe.
3. **Combine:** merges the chunks, then applies RMSNorm × sigmoid(gate) to BF16.

Per-shape tuning (block size, warps, stages, split cap, page-index prefetch on decode) took it from
v1 1.21x to **v7 1.434x scored**. Sliding-window calls run 1.6-2.2x faster than SGLang's.

**Lessons:** a runtime branch inside the key loop stops Triton pipelining the loads (that version scored 0.84x).
FP8 P @ V was no faster, because Triton's operand conversion ate the gain. The remaining gap is in the
page-size-1 gather on full layers, which runs at about 65% of peak.

## 3. Routed experts: `muse_moe_w4a16` ([moe_w4a16/](moe_w4a16/))

Top-8 of 256 experts, SwiGLU, NVFP4 weights with BF16 activations, and a norm on each expert's output.

**What the submission does** ([muse_moe_w4a16.py](best_submissions/muse_moe_w4a16.py)): it keeps SGLang's Marlin
W4A16 GEMMs and replaces the glue around them:
- **One Triton finalize kernel** for the post-expert RMSNorm × (1 + w) and top-k weighted sum. Up to
  16 tokens, SGLang falls back to about 14 PyTorch kernels here (34-43 µs of the call).
- **One Triton align kernel** that writes Marlin's sorted-ids/block layout directly, replacing the
  expert-map index, its copy and `moe_align_block_size`. It uses block size 16 at 9-32 tokens.

Result: **1.179x**, with 1.57x at batch size 1. Large batches stay at parity because Marlin dominates there.

**Explored and rejected (each correct, but slower than Marlin):**
- A Triton grouped GEMM with an exact inline-PTX E2M1 → BF16 dequant, split-K and a tile-major
  repack. It got to within 1.10-1.26x of Marlin but never ahead ([moe_triton*.py](moe_w4a16/)).
- A hand-written CUDA `mma.sync` grouped kernel ([moe_cuda.cu](moe_w4a16/moe_cuda.cu)): correct on
  its first GPU run, but 1.4-2.3x slower.
- Library W4A16 paths (FlashInfer `mm_bf16_fp4`, CUTLASS mixed-input): none supports NVFP4 with
  BF16 activations on sm90.

## 4. Whole decoder layer: `muse_decode_layer` ([decode_layer/](decode_layer/))

This problem combines the pieces above. It is SGLang's layer with the stages swapped one at a time,
each swap measured:

| Version | Change | Speedup |
|---|---|---|
| v1 | bf16_proj + attention-decode kernels dropped in; dead 512 MB RoPE table removed | 1.089x |
| v2 | + fused post-expert norm and weighted sum | 1.172x |
| v3 | + router as one fused hi/lo Triton kernel, overlapped with the pooler on a side stream | 1.197x |
| v4 | + one kernel for the pre-expert RMSNorm and the slot → local-expert map | 1.254x |
| v5 | + both norm/residual boundaries as Triton kernels | 1.257x |
| v7 | + Marlin block layout from one Triton kernel, block size 16 at 9-32 tokens | **1.271x** |

(v6 folded routing into the pre-expert kernel but moved it off the side stream onto the
critical path, so it gained nothing.) The v7 file is
[muse_decode_layer.py](best_submissions/muse_decode_layer.py).

## 5. Prefill layer: `muse_prefill_layer` ([prefill_layer/](prefill_layer/))

The same layer at prefill sizes (753-token turns on an 80K context, and 16K-token first-turn
chunks). Here 65-69% of the time is the MoE GEMM at large M, where Marlin reaches only about 17-27% of
BF16 tensor-core peak.

**What the submission does** ([muse_prefill_layer.py](best_submissions/muse_prefill_layer.py)): for ≥ 8,192 tokens it
dequantizes the NVFP4 experts exactly to BF16 in scratch memory, on a side stream that overlaps
attention and the projections. It then scatters tokens by expert and runs `torch._grouped_mm` (about 750
TFLOPS), followed by a grouped SiLU-mul and finalize. Smaller blocks stay on Marlin, where the
1.4 ms dequant doesn't pay for itself.

Result: v1 1.000x (parity) → **v3 1.133x submitted**, with 1.28-1.50x on the 16K-token shapes.
The weights stay NVFP4 in memory, and the dequant runs inside the timed call.

---

## Mathematical formulation

These are the functions each submission computes, taken from the judge's reference code, plus the
reformulations the kernels use to compute them faster. All accumulation is in FP32.

### Common building blocks

```math
\mathrm{RMSNorm}_\epsilon(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}},
\qquad \sigma(z) = \frac{1}{1+e^{-z}},
\qquad \mathrm{SiLU}(z) = z\,\sigma(z)
```

$`\epsilon = 10^{-5}`$ unless noted. "Norm with weight $`w`$" means $`\mathrm{RMSNorm}(x)\odot(1+w)`$.

### 1. Dense projections (`muse_bf16_proj`)

A plain GEMM, $`X \in \mathbb{R}^{m\times k}`$, $`W \in \mathbb{R}^{n\times k}`$:

```math
Y = X W^\top, \qquad y_{ij} = \sum_{l=1}^{k} x_{il}\, w_{jl}
```

**Split-K.** K is cut into $`S`$ slices $`\mathcal{K}_1,\dots,\mathcal{K}_S`$, one per work unit, so every SM
streams a share of $`W`$; the partial sums meet in an FP32 workspace by atomic adds:

```math
y_{ij} = \sum_{s=1}^{S} \underbrace{\sum_{l \in \mathcal{K}_s} x_{il}\, w_{jl}}_{\text{one work unit}}
```

**Swap-AB.** For small $`m`$ the kernel computes the transpose, so the wide weight tile is the
tensor-core operand that fills the MMA: $`Y^\top = W X^\top`$.

**Router (FP32 weights) as a BF16 hi + lo pair**, split once in `prepare()`:

```math
W_{hi} = \mathrm{bf16}(W), \qquad W_{lo} = \mathrm{bf16}(W - W_{hi}), \qquad
Y \approx X W_{hi}^\top + X W_{lo}^\top
```

Both products go through the tensor cores into one FP32 accumulator in a single pass over the
weights. The pair carries about 16 mantissa bits, well inside the router's $`10^{-4}`$ error bound.
When the input is FP32 too (the decode layer, up to 8 tokens), it is split the same way and the
negligible $`x_{lo}W_{lo}`$ term is dropped:
$`Y \approx x_{hi}W_{hi}^\top + x_{hi}W_{lo}^\top + x_{lo}W_{hi}^\top`$.

**m = 1 (GEMV).** $`y_j = \sum_l x_l w_{jl}`$, with each warp summing one row segment and the warps
of a row reduced in shared memory. The router's FP32 weights are used directly here.

### 2. Gated attention decode (`muse_attention_decode`)

Per token at position $`p`$ and query head $`h`$ (KV head $`g`$ of its group), head dim $`d = 64`$:

```math
\hat q = c\cdot\mathrm{RMSNorm}(q), \quad \hat k = \mathrm{RMSNorm}(k), \qquad c = 5.473
```

**RoPE** (sliding layers only, interleaved/GPT-J form), $`\theta = 500{,}000`$:

```math
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos\phi_i & -\sin\phi_i \\ \sin\phi_i & \cos\phi_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}, \qquad \phi_i = p\,\theta^{-2i/d}
```

**FP8 cache write:** $`\tilde K = \mathrm{E4M3}(\hat k / s_k)`$, $`\tilde V = \mathrm{E4M3}(v / s_v)`$.

**Attention** over the visible keys $`t \in [t_0, p]`$, with $`t_0 = \max(0, p - 2048 + 1)`$ on sliding
layers and $`0`$ on full layers (causal inside an 8-token verify block):

```math
\ell_t = \frac{\hat q \cdot k_t}{\sqrt d}, \qquad
o = \sum_{t=t_0}^{p} \frac{e^{\ell_t}}{\sum_{u} e^{\ell_u}}\, v_t, \qquad
\text{out} = \mathrm{RMSNorm}(o) \odot \sigma(\text{gate})
```

**FP8 query (as FA3 does):** the query is rounded after the norm and $`c`$ moves into the softmax scale,
$`\ell_t = \mathrm{E4M3}(\mathrm{RMSNorm}(q))\cdot k_t \cdot c/\sqrt d`$.

**Split-KV with online softmax.** Each program handles a key chunk $`j`$ and keeps the running max,
sum and unnormalized output:

```math
m_j = \max_{t\in j} \ell_t, \qquad
L_j = \sum_{t\in j} e^{\ell_t - m_j}, \qquad
O_j = \sum_{t\in j} e^{\ell_t - m_j}\, v_t
```

The combine kernel merges the chunks exactly:

```math
m = \max_j m_j, \qquad
o = \frac{\sum_j e^{m_j - m}\, O_j}{\sum_j e^{m_j - m}\, L_j}
```

### 3. Routed experts, NVFP4 (`muse_moe_w4a16`)

**NVFP4 weights:** two 4-bit E2M1 codes per byte, one E4M3 scale $`s`$ per 16 values, one FP32
global scale $`g`$ per expert:

```math
W_{r,l} = e_{r,l}\cdot s_{r,\lfloor l/16\rfloor}\cdot g, \qquad
e \in \{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}
```

**Per token** $`x`$ with its top-8 experts $`\mathcal{E}`$ and routing weights $`w_e`$ ($`\sum_e w_e = 1`$),
$`D = 4096`$:

```math
h = x\,W_{13,e}^\top = [\,h_{gate} \mid h_{up}\,], \qquad
a = \mathrm{SiLU}(h_{gate}) \odot h_{up}, \qquad
y_e = a\,W_{2,e}^\top
```

```math
\hat y_e = \mathrm{RMSNorm}_{10^{-8}}\big(y_e \odot (1 + w_{post})\big), \qquad
\text{out} = \sum_{e \in \mathcal{E}\,\cap\,\text{local}} w_e\, \hat y_e
```

The $`(1+w_{post})`$ scale is applied *before* the normalization, in FP32. My finalize kernel computes
$`\hat y_e`$ and the weighted sum in one launch.

### 4. Whole decoder layer (`muse_decode_layer`)

**Norm-preserving residual** with a learned gate $`\gamma`$:

```math
z = \gamma / 0.3, \qquad \beta = \sigma(z), \qquad
\alpha = \sqrt{\max\big(\sigma(-z)\,(1+\beta),\ 10^{-3}\big)}
```

**The layer**, from residual stream $`s`$ and attention input $`x`$:

```math
\begin{aligned}
a      &= \mathrm{Attn}(x)\, W_o^\top \\
s_1    &= \alpha_1\, s + \beta_1\, \mathrm{RMSNorm}_{10^{-8}}(a) \\
h      &= \mathrm{RMSNorm}(s_1) \odot (1 + w_{\text{pre-ffn}}) \\
r      &= \sigma\big(0.1768\cdot h\, W_r^\top\big), \qquad \mathcal{E} = \operatorname{top8}(r + b) \\
w_e    &= r_e \Big/ \Big(\textstyle\sum_{e'\in\mathcal{E}} r_{e'} + 10^{-15}\Big) \\
p      &= \mathrm{RMSNorm}(h\, W_{pool}^\top) \odot (1 + w_{\text{pre-expert}}) \\
f      &= \big(\mathrm{MoE}(p)\, W_{unpool}^\top\big) \odot (1 + w_{\text{post-ffn}}) \\
s_2    &= \alpha_2\, s_1 + \beta_2\, \mathrm{RMSNorm}_{10^{-8}}(f) \\
x_{next} &= \mathrm{RMSNorm}(s_2) \odot (1 + w_{\text{next-attn}})
\end{aligned}
```

$`\mathrm{Attn}`$ is section 2, $`\mathrm{MoE}`$ is section 3, and the projections are section 1. The
bias $`b`$ only chooses the experts; the weights $`w_e`$ use the unbiased scores. Up to 8 tokens this
GPU computes only the first $`8192/8 = 1024`$ columns of $`f`$. My kernels fuse the norm/residual
lines and the router into single launches; the math is unchanged.

### 5. Prefill layer (`muse_prefill_layer`)

The same layer as section 4. The change is how the experts are computed at ≥ 8,192 tokens. The
global scales are factored out of the GEMMs, so the dequantized weights are exact in BF16:

```math
\tilde W = e \cdot s \ \ (\text{exact in BF16}), \qquad
H_e = g_{13}\,\big(X_e \tilde W_{13,e}^\top\big), \qquad
Y_e = g_{2}\,\big(A_e \tilde W_{2,e}^\top\big)
```

Here $`X_e`$ holds all tokens routed to expert $`e`$, so the whole step is one grouped GEMM
(`torch._grouped_mm`) per projection, followed by the SiLU-mul and the same finalize as section 3.

### How the judge scores it

```math
T_{\text{load}} = \sum_{\text{shapes}} \text{weight}\times t_{\text{shape}}, \qquad
\text{speedup} = \Big(\prod_{\text{loads}} \frac{T^{\text{SGLang}}_{\text{load}}}{T^{\text{mine}}_{\text{load}}}\Big)^{1/N_{\text{loads}}}
```

The floor for each shape is $`\max(\text{bytes}/4.8\,\text{TB/s},\ \text{FLOPs}/989\,\text{TFLOP/s})`$.
Correctness is relative L2 against an FP64 reference: $`\lVert \text{out} - \text{ref}\rVert_2 / \lVert \text{ref}\rVert_2`$.

---

## Repository layout

```
.
├── README.md                  # this file
├── .gitignore                 # keeps TASK.md, AGENTS.md, intent.md and working submission*.py out of git
├── best_submissions/          # the best submission of every problem, in one place
├── bf16_proj/                 # muse_bf16_proj
├── attn_decode/               # muse_attention_decode (my working copy)
├── muse_attention_decode/     # untouched `okc init` starter for the same problem
├── moe_w4a16/                 # muse_moe_w4a16
├── decode_layer/              # muse_decode_layer
├── prefill_layer/             # muse_prefill_layer
└── 10t_model/                 # side study: sizing a 10T-parameter model on Cerebras wafers
```

Inside each problem folder:

| File | Purpose |
|---|---|
| `LOG.md` | Every experiment: hypothesis, judge job ID, result, and why it was kept or dropped |
| `baseline.py` | SGLang's implementation, the one I'm timed against |
| `reference.py`, `checks.py` | The judge's reference computation and correctness checks |
| `probe*.py`, `tune*.py` | Exploration and tuning scripts run on the judge via `okc check` |
| `*.cu`, `moe_triton*.py` | Native CUDA and alternative Triton prototypes |
| `profiles/` | Judge outputs per job (`.json` result, `.log` output, `.txt` profile) |

Each folder's working `submission.py` (and older variants) stays local; the final version of each
is in [best_submissions/](best_submissions/). The problem statements (`TASK.md`) and agent notes
(`AGENTS.md`), plus my local planning notes (`intent.md`), are not in the repository either (see [.gitignore](.gitignore)). To get a problem's
statement, run `okc init <problem>` (for example `okc init muse_bf16_proj`) or read it on
[openkernelchallenge.dev](https://openkernelchallenge.dev/).

## Reproducing

You need the `okc` CLI and an account that has accepted the rules (`okc login`, run by a person).
No local GPU is needed: everything runs on the judge's H200.

```bash
okc check best_submissions/muse_bf16_proj.py --json    # tests + timing, kept off the leaderboard
okc submit best_submissions/muse_bf16_proj.py --json   # scored run
okc leaderboard muse_bf16_proj --json
```

The judge environment: torch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0, H200 SXM. Locally I checked
correctness without a GPU using `TRITON_INTERPRET=1` against the judge's `reference.py`.

## Rules I kept to

These solutions aim for real speedups that could be upstreamed, not tricks against the judge:
- Weights are repacked only losslessly and keep their precision (BF16 stays BF16, NVFP4 stays NVFP4).
  The router's BF16 hi + lo split is the one the rules allow.
- No caching of outputs or input-derived data, and no work moved outside the timed call.
- Everything is CUDA-graph safe: no host syncs, no branching on input values.
- Correctness is checked against the FP64 reference, including stress runs for race conditions.

## Side study: 10T model on Cerebras ([10t_model/](10t_model/))

This is not part of the kernel challenge. It sizes a 10-trillion-parameter model on Cerebras WSE-3T wafers
using the method from Cerebras' "Deploying Trillion-parameter Frontier Models" talk: about 128-190 wafers,
with the trade-off between wafer count and KV-cache room. See [10t_model/README.md](10t_model/README.md)
and `python3 10t_model/calc.py`.
