# 10T-parameter model on Cerebras wafers

Sizing for a 10-trillion-parameter model, using the method in the Cerebras talk
"Deploying Trillion-parameter Frontier Models" (Sean Lie, https://youtu.be/efqQMmvSsVE)
and its companion post
https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras

`python3 calc.py` prints everything; `python3 calc.py --params 10e12 --layers 160` for other shapes.

## Method (from the post)

- Weights at 4-bit, KV cache at 8-bit, 256k-token requests.
- Each WSE-3T wafer holds 44 GB of SRAM. Pipeline parallel: whole layers stay on a wafer, only activations move (~2 us per hop).
- Linear-attention layers have a fixed-size state, so pack as many per wafer as fit.
- Dense (full-attention) layers take one wafer each; the SRAM left over holds KV cache.

Check: Kimi K3 (3T, 93 layers = 69 linear + 24 dense) -> 1,500 GB, 16.1 GB/layer,
35 + 24 = **59 wafers**, as the post says.

## 10T results (4-bit weights = 5,000 GB)

| Layers (linear + dense) | GB/layer | Wafers | KV room per dense wafer | 256k requests at once (MLA KV) | Hop latency/token |
|---|---|---|---|---|---|
| 93 (70 + 23)  | 53.8 | 186 (each layer split over 2) | 34.2 GB | 226 | 370 us |
| **128 (96 + 32)** | **39.1** | **128** | **4.9 GB** | **32** | **254 us** |
| 192 (144 + 48) | 26.0 | 192 | 18.0 GB | 118 | 382 us |

Variants on 128 layers: GQA-8 KV instead of MLA -> 9 requests; 8-bit weights -> 256 wafers, 65 requests.

Takeaways:
- Minimum is ~114 wafers just to hold 5 TB (5,000 / 44); a realistic deployment is ~128-190 wafers, about 2-3x Kimi K3's 59.
- At 10T a layer is ~26-54 GB, close to or above one wafer, so linear layers no longer pair up two per wafer as they do for K3.
- The fewest wafers (128 layers) leave almost no KV room, so concurrency collapses to ~32. Deeper models (192 layers) or splitting layers buy back KV space at the cost of more wafers.

## Assumptions (not in the post)

- No public 10T config: same ~3:1 linear:dense mix as K3, depths 93 / 128 / 192.
- KV per token per dense layer: MLA 576 B (512 latent + 64 rope, as in DeepSeek V3 / Kimi K2) or GQA 8 heads x 128 dims = 2,048 B, at 8-bit.
- Weights split evenly over layers (embeddings, norms, routers ignored).
- Concurrency counts KV only; activations, linear-attention state and scratch are ignored, so real numbers are lower.
- No compute or bandwidth figures in the post, so no tokens/s estimate; hop latency is only the network part of per-token latency.
