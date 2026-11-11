"""Wafer-deployment sizing for trillion-parameter models, after Cerebras'
"How we deploy trillion-parameter frontier models" (Kimi K3 worked example).

Method (from the post):
  weights  = params x bits / 8, split evenly over layers
  linear-attention layers keep a fixed-size state -> pack as many per wafer as fit
  dense (full-attention) layers need KV cache -> one per wafer, the rest of SRAM is KV
  pipeline parallel: activations hop between wafers (~2 us per hop)

Run:  python3 calc.py            # Kimi K3 check + the 10T scenarios
      python3 calc.py --params 10e12 --layers 128 --dense-every 4
"""
import argparse
import math

GB = 1e9
SRAM_GB = 44.0      # WSE-3T on-chip SRAM per wafer (from the post)
HOP_US = 2.0        # RDMA-over-Ethernet hop between wafers (from the post)

# KV bytes per token per dense layer at 8-bit. The post doesn't give K3's
# dims; these are assumptions: MLA (512 latent + 64 rope, as in DeepSeek V3 /
# Kimi K2) or GQA with 8 KV heads x 128 dims, K and V.
KV_BYTES_PER_TOKEN = {"mla": (512 + 64) * 1, "gqa8": 2 * 8 * 128 * 1}


def size(name, params, layers, dense, weight_bits=4, kv="mla", ctx=256 * 1024):
    linear = layers - dense
    weights_gb = params * weight_bits / 8 / GB
    layer_gb = weights_gb / layers

    if layer_gb <= SRAM_GB:
        per_wafer = int(SRAM_GB // layer_gb)              # linear layers per wafer
        linear_wafers = math.ceil(linear / per_wafer) if linear else 0
        wafers_per_dense = 1
        kv_room_gb = SRAM_GB - layer_gb                   # per dense layer
    else:
        # a layer is bigger than one wafer: split it across several
        per_wafer = 0
        span = math.ceil(layer_gb / SRAM_GB)
        linear_wafers = linear * span
        # one extra wafer per dense layer if the split leaves too little for KV
        wafers_per_dense = span if span * SRAM_GB - layer_gb >= 8 else span + 1
        kv_room_gb = wafers_per_dense * SRAM_GB - layer_gb
    dense_wafers = dense * wafers_per_dense
    wafers = linear_wafers + dense_wafers

    kv_req_gb = KV_BYTES_PER_TOKEN[kv] * ctx / GB         # one request, one dense layer
    concurrency = int(kv_room_gb // kv_req_gb)
    kv_total_gb = kv_req_gb * dense                       # one request, whole model

    print(f"\n== {name}")
    print(f"  params {params / 1e12:.1f}T @ {weight_bits}-bit -> weights {weights_gb:,.0f} GB")
    print(f"  layers {layers} ({linear} linear, {dense} dense) -> {layer_gb:.1f} GB per layer")
    if per_wafer:
        print(f"  linear wafers: {linear} / {per_wafer} per wafer -> {linear_wafers}")
    else:
        print(f"  layer > {SRAM_GB:.0f} GB wafer: each layer spans {span} wafers -> linear {linear_wafers}")
    print(f"  dense wafers:  {dense} x {wafers_per_dense} -> {dense_wafers}, KV room {kv_room_gb:.1f} GB per dense layer")
    print(f"  TOTAL WAFERS:  {wafers}   (SRAM {wafers * SRAM_GB:,.0f} GB, {weights_gb / (wafers * SRAM_GB):.0%} weights)")
    print(f"  KV ({kv}, 8-bit, {ctx // 1024}k ctx): {kv_req_gb * 1000:.0f} MB per dense layer, "
          f"{kv_total_gb:.1f} GB per request")
    print(f"  concurrent {ctx // 1024}k requests (KV-bound): {concurrency}")
    print(f"  pipeline hop latency per token: {wafers - 1} hops x {HOP_US:.0f} us = {(wafers - 1) * HOP_US:.0f} us")
    return wafers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--dense-every", type=int, default=4, help="1 dense layer per N layers (K3: ~4)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--kv", choices=KV_BYTES_PER_TOKEN, default="mla")
    ap.add_argument("--ctx", type=int, default=256 * 1024)
    a = ap.parse_args()

    if a.params:
        layers = a.layers or 128
        size("custom", a.params, layers, layers // a.dense_every, a.bits, a.kv, a.ctx)
        return

    # check: the post's Kimi K3 numbers (93 layers, 24 dense) -> 59 wafers
    assert size("Kimi K3 (check vs post: 59 wafers)", 3e12, 93, 24) == 59

    # 10T: no public config; same 3:1 linear:dense mix as K3, three depths
    for layers in (93, 128, 192):
        size(f"10T, {layers} layers", 10e12, layers, layers // 4)
    size("10T, 128 layers, GQA-8 KV", 10e12, 128, 32, kv="gqa8")
    size("10T, 128 layers, 8-bit weights", 10e12, 128, 32, weight_bits=8)


if __name__ == "__main__":
    main()
