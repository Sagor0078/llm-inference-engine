# okc: problem=muse_prefill_layer
"""Probe P2, not a submission: prefill MoE as exact NVFP4 -> BF16 dequant (in-call, to scratch) + torch._grouped_mm (grouped
BF16 GEMM, device offsets) + per-expert-scaled SiLU-mul + finalize, vs SGLang's Marlin path, at 753 / 3012 / 16384 tokens."""

import math
import time

import triton
import triton.language as tl
from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl



HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)





def _route(tokens, seed, dev):
    """Skewed top-8 routing like the layer's (a few hot experts), with this GPU's experts 0-31 the busiest."""
    g = torch.Generator(device=dev).manual_seed(seed)
    scores = torch.rand((tokens, 256), generator=g, device=dev) + torch.rand((256,), generator=g, device=dev).pow(6) * 4
    routed = scores.topk(8, -1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = ((ids - 32 * pairs.argmax()) % 256).to(torch.int32)
    return ids, (routed.values / routed.values.sum(-1, keepdim=True)).float()



@triton.jit
def _dequant_kernel(p_ptr, sf_ptr, out_ptr, n_bytes, BLOCK: tl.constexpr):
    """Packed E2M1 (2 per byte, low nibble first) x E4M3 block scale (16 values) -> bf16, exact (global scale not applied)."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n_bytes
    b = tl.load(p_ptr + i, mask=ok, other=0).to(tl.int32)
    s = tl.load(sf_ptr + i // 8, mask=ok, other=0.0).to(tl.float32)
    for half in tl.static_range(2):
        c = (b >> (4 * half)) & 15
        mag = c & 7
        big = ((mag + 252) << 22).to(tl.float32, bitcast=True)
        v = tl.where(mag >= 2, big, mag.to(tl.float32) * 0.5)
        v = tl.where((c & 8) != 0, -v, v) * s
        tl.store(out_ptr + 2 * i + half, v.to(tl.bfloat16), mask=ok)


@triton.jit
def _group_kernel(ids_ptr, rows_ptr, offs_ptr, n_slots, COUNT: tl.constexpr, NS: tl.constexpr, ALIGN: tl.constexpr):
    """Slots grouped by local expert (ascending slot within each), each group padded to ALIGN rows: rows_ptr[r] = slot or -1;
    offs_ptr[e] = end row of expert e's group (cumulative, as torch._grouped_mm's offs)."""
    s = tl.arange(0, NS)
    s_ok = s < n_slots
    e = tl.load(ids_ptr + s, mask=s_ok, other=-1)
    local = s_ok & (e >= 0) & (e < COUNT)
    e = tl.where(local, e, COUNT)
    counts = tl.histogram(e, 64)
    ex = tl.arange(0, 64)
    counts = tl.where(ex < COUNT, counts, 0)
    padded = (counts + ALIGN - 1) // ALIGN * ALIGN
    end = tl.cumsum(padded, 0)
    start = end - padded
    first = tl.cumsum(counts, 0) - counts
    key = tl.sort(e * NS + s)
    es = key // NS
    ss = key % NS
    pos = tl.arange(0, NS)
    dest = tl.gather(start, es, 0) + pos - tl.gather(first, es, 0)
    keep = (es < COUNT) & (ss < n_slots)
    tl.store(rows_ptr + dest, ss.to(tl.int32), mask=keep)
    tl.store(offs_ptr + ex, end.to(tl.int32), mask=ex < COUNT)


def prepare(weights):
    t0 = time.time()
    state = {"bank": _marlin_bank(weights),
             "post_expert_w": (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16)}
    dev = weights["w13"].device
    lines = [f"has _grouped_mm: {hasattr(torch, '_grouped_mm')}"]
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)

    def bench(fn, reps=5):
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        ts = []
        for _ in range(reps):
            torch.sum(flush, dim=0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gr.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    E, I2, Hh = weights["w13"].shape
    H = Hh * 2
    I = I2 // 2
    w13b = torch.empty((E, I2, H), device=dev, dtype=torch.bfloat16)
    w2b = torch.empty((E, H, I), device=dev, dtype=torch.bfloat16)
    n13 = weights["w13"].numel()
    n2 = weights["w2"].numel()

    def dequant_all():
        _dequant_kernel[(triton.cdiv(n13, 4096),)](weights["w13"], weights["w13_sf"], w13b, n13, BLOCK=4096, num_warps=8)
        _dequant_kernel[(triton.cdiv(n2, 4096),)](weights["w2"], weights["w2_sf"], w2b, n2, BLOCK=4096, num_warps=8)

    dequant_all()
    torch.cuda.synchronize()
    lines.append(f"dequant all 32 experts (w13 + w2 -> bf16): {bench(dequant_all):.1f} us")
    gs13 = weights["w13_gs"].float()
    gs2 = weights["w2_gs"].float()
    for tokens in (753, 3012, 16384):
        if time.time() - t0 > 250:
            lines.append("TIME BUDGET HIT")
            break
        x = torch.randn((tokens, H), device=dev).to(torch.bfloat16)
        ids, wts = _route(tokens, tokens, dev)
        n_slots = tokens * 8
        max_rows = n_slots + E * 16
        marlin = lambda: _experts(x, wts, ids, state)
        ref = marlin().float()
        tm = bench(marlin)
        rows = torch.full((max_rows,), -1, dtype=torch.int32, device=dev)
        offs = torch.empty((E,), dtype=torch.int32, device=dev)

        def grouped():
            rows.fill_(-1)
            _group_kernel[(1,)](ids, rows, offs, n_slots, COUNT=E, NS=triton.next_power_of_2(n_slots), ALIGN=16,
                                num_warps=8)
            ok = rows >= 0
            src = torch.where(ok, rows // 8, 0).long()
            a = x[src] * ok[:, None].to(x.dtype)                                  # [max_rows, H] (pads zero)
            h = torch._grouped_mm(a, w13b.transpose(1, 2), offs=offs)             # [max_rows, 2I]
            expert = torch.searchsorted(offs, torch.arange(max_rows, device=dev, dtype=torch.int32), right=True).clamp(max=E - 1)
            g = h[:, :I].float() * gs13[expert, None]
            u = h[:, I:].float() * gs13[expert, None]
            inter = (g * torch.sigmoid(g) * u).to(torch.bfloat16)
            y = torch._grouped_mm(inter, w2b.transpose(1, 2), offs=offs).float() * gs2[expert, None]
            z = y * (1.0 + weights["post_expert_norm"].float())
            z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-8)
            out = torch.zeros((tokens, H), device=dev, dtype=torch.float32)
            slot = torch.where(ok, rows, 0).long()
            wrow = torch.where(ok, wts.flatten()[slot], 0.0)
            out.index_add_(0, slot // 8, z * wrow[:, None])
            return out.to(torch.bfloat16)

        try:
            got = grouped().float()
            err = float((got - ref).norm() / ref.norm())
            tg = bench(grouped)
            tg_dq = bench(lambda: (dequant_all(), grouped()))
            lines.append(f"tokens {tokens}: marlin {tm:.1f} us | grouped (excl. dequant) {tg:.1f} us, incl. dequant {tg_dq:.1f} us, rel err vs marlin {err:.2e} [{time.time() - t0:.0f}s]")
        except Exception as e:  # noqa: BLE001
            lines.append(f"tokens {tokens}: marlin {tm:.1f} | grouped FAILED {type(e).__name__}: {str(e)[:300]}")
        print(lines[-1], flush=True)
        torch.cuda.empty_cache()
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
