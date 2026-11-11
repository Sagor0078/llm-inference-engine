# okc: problem=muse_moe_w4a16
"""Probe M1, not a submission: Triton NVFP4 grouped-GEMM MoE vs SGLang's Marlin path per batch size, with a
tile sweep; inputs built like the judge's generator (popularity-skewed routing, busiest-GPU rotation).
Runs in prepare(), then stops the run with the table."""

import math
import time
from typing import TypedDict

import torch
import triton
import triton.language as tl
import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


@triton.jit
def _e2m1_scaled(code, sf):
    """E2M1 code (int32, 4 bits) times its block scale (fp32) -> bf16, exactly."""
    mag = code & 7
    # mag >= 2: exponent (mag >> 1) - 1, mantissa (mag & 1) / 2, i.e. FP32 bits (mag + 252) << 22.
    big = ((mag + 252) << 22).to(tl.float32, bitcast=True)
    v = tl.where(mag >= 2, big, mag.to(tl.float32) * 0.5)
    v = tl.where((code & 8) != 0, -v, v)
    return (v * sf).to(tl.bfloat16)


@triton.jit
def _align_kernel(ids_ptr, sorted_ptr, block_expert_ptr, local_start, n_slots, max_blocks,
                  COUNT: tl.constexpr, BINS: tl.constexpr, BM: tl.constexpr, NS: tl.constexpr, NB: tl.constexpr):
    """One program: the slots routed to local experts, grouped by expert into BM-row blocks.
    sorted[b * BM + r] = slot (token * 8 + k) or -1; block_expert[b] = local expert or -1."""
    s = tl.arange(0, NS)
    s_ok = s < n_slots
    e = tl.load(ids_ptr + s, mask=s_ok, other=-1) - local_start
    local = s_ok & (e >= 0) & (e < COUNT)
    e = tl.where(local, e, COUNT)  # non-local last
    counts = tl.histogram(e, BINS)  # BINS: a power of two > COUNT
    ex = tl.arange(0, BINS)
    counts = tl.where(ex < COUNT, counts, 0)
    blocks = (counts + BM - 1) // BM
    block_end = tl.cumsum(blocks, 0)
    block_start = block_end - blocks
    # Sort slots by (expert, slot); rank within the expert = sorted position - expert's first position.
    key = e * NS + s
    key_sorted = tl.sort(key)
    e_sorted = key_sorted // NS
    s_sorted = key_sorted % NS
    first = tl.cumsum(counts, 0) - counts  # sorted position of each expert's first slot
    pos = tl.arange(0, NS)
    # rank = pos - first[e_sorted]; dest row = block_start[e_sorted] * BM + rank
    first_g = tl.gather(first, e_sorted, 0)
    bstart_g = tl.gather(block_start, e_sorted, 0)
    dest = bstart_g * BM + (pos - first_g)
    keep = (e_sorted < COUNT) & (s_sorted < n_slots)
    # Pad rows first (-1), then the real slots.
    rows = tl.arange(0, NB * BM)
    tl.store(sorted_ptr + rows, tl.full([NB * BM], -1, tl.int32), mask=rows < max_blocks * BM)
    tl.debug_barrier()
    tl.store(sorted_ptr + dest, s_sorted.to(tl.int32), mask=keep)
    b = tl.arange(0, NB)
    be = tl.sum(tl.where((b[:, None] >= block_start[None, :]) & (b[:, None] < block_end[None, :]),
                         ex[None, :], 0), 1)
    used = b < tl.max(block_end, 0)
    tl.store(block_expert_ptr + b, tl.where(used, be, -1).to(tl.int32), mask=b < max_blocks)


@triton.jit
def _gemm1_kernel(x_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, inter_ptr,
                  H, I, TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """inter[slot, n] = silu(x[tok] @ gate[e, n]^T) * (x[tok] @ up[e, n]^T) for BM slots x BN columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    tok = tl.where(row_ok, slot // TOPK, 0)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    e64 = e.to(tl.int64)
    wg = w_ptr + (e64 * (2 * I) + n)[:, None] * (H // 2)          # gate rows
    wu = w_ptr + (e64 * (2 * I) + I + n)[:, None] * (H // 2)      # up rows
    sg = sf_ptr + (e64 * (2 * I) + n)[:, None] * (H // 16)
    su = sf_ptr + (e64 * (2 * I) + I + n)[:, None] * (H // 16)
    xrow = x_ptr + tok[:, None].to(tl.int64) * H
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, H, BK):
        xe = tl.load(xrow + k0 + 2 * i[None, :], mask=row_ok[:, None], other=0.0)
        xo = tl.load(xrow + k0 + 2 * i[None, :] + 1, mask=row_ok[:, None], other=0.0)
        pg = tl.load(wg + k0 // 2 + i[None, :]).to(tl.int32)
        pu = tl.load(wu + k0 // 2 + i[None, :]).to(tl.int32)
        fg = tl.load(sg + k0 // 16 + i[None, :] // 8).to(tl.float32)
        fu = tl.load(su + k0 // 16 + i[None, :] // 8).to(tl.float32)
        acc_g = tl.dot(xe, tl.trans(_e2m1_scaled(pg & 15, fg)), acc_g)
        acc_g = tl.dot(xo, tl.trans(_e2m1_scaled(pg >> 4, fg)), acc_g)
        acc_u = tl.dot(xe, tl.trans(_e2m1_scaled(pu & 15, fu)), acc_u)
        acc_u = tl.dot(xo, tl.trans(_e2m1_scaled(pu >> 4, fu)), acc_u)
    gs = tl.load(gs_ptr + e)
    g = acc_g * gs
    u = acc_u * gs
    h = g * tl.sigmoid(g) * u
    dst = inter_ptr + tl.where(row_ok, slot, 0)[:, None].to(tl.int64) * I + n[None, :]
    tl.store(dst, h.to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit
def _gemm2_kernel(inter_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, y_ptr,
                  H, I, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """y[slot, n] = inter[slot] @ w2[e, n]^T for BM slots x BN columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    srow = tl.where(row_ok, slot, 0).to(tl.int64)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    e64 = e.to(tl.int64)
    wp = w_ptr + (e64 * H + n)[:, None] * (I // 2)
    sp = sf_ptr + (e64 * H + n)[:, None] * (I // 16)
    arow = inter_ptr + srow[:, None] * I
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, I, BK):
        ae = tl.load(arow + k0 + 2 * i[None, :], mask=row_ok[:, None], other=0.0)
        ao = tl.load(arow + k0 + 2 * i[None, :] + 1, mask=row_ok[:, None], other=0.0)
        p = tl.load(wp + k0 // 2 + i[None, :]).to(tl.int32)
        f = tl.load(sp + k0 // 16 + i[None, :] // 8).to(tl.float32)
        acc = tl.dot(ae, tl.trans(_e2m1_scaled(p & 15, f)), acc)
        acc = tl.dot(ao, tl.trans(_e2m1_scaled(p >> 4, f)), acc)
    y = acc * tl.load(gs_ptr + e)
    tl.store(y_ptr + srow[:, None] * H + n[None, :], y.to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit
def _finalize_kernel(y_ptr, ids_ptr, wts_ptr, norm_w_ptr, out_ptr, local_start, H,
                     COUNT: tl.constexpr, TOPK: tl.constexpr, BH: tl.constexpr, EPS: tl.constexpr):
    """out[t] = sum over t's local slots of weight * rmsnorm(y * (1 + w))."""
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    h_ok = h < H
    w1 = 1.0 + tl.load(norm_w_ptr + h, mask=h_ok, other=0.0).to(tl.float32)
    out = tl.zeros([BH], tl.float32)
    for k in range(TOPK):
        e = tl.load(ids_ptr + t * TOPK + k) - local_start
        if (e >= 0) & (e < COUNT):
            z = tl.load(y_ptr + (t * TOPK + k).to(tl.int64) * H + h, mask=h_ok, other=0.0).to(tl.float32) * w1
            inv = tl.rsqrt(tl.sum(z * z, 0) / H + EPS)
            out += tl.load(wts_ptr + t * TOPK + k) * z * inv
    tl.store(out_ptr + t * H + h, out.to(tl.bfloat16), mask=h_ok)


def block_m(m: int) -> int:
    """Rows per expert block from the token count (shape-only)."""
    return 16 if m <= 16 else 32 if m <= 64 else 64


def moe_forward(x, topk_ids, topk_wts, w13, w13_sf, w13_gs, w2, w2_sf, w2_gs, norm_w, local_start: int,
                count: int, topk: int = 8, bn1: int = 64, bn2: int = 64, bk: int = 64, warps: int = 4,
                stages: int = 3, eps: float = 1e-8, bm: int = 0):
    """w13 [count, 2I, H/2] uint8, w13_sf [count, 2I, H/16] e4m3, w13_gs [count] fp32 (the local experts);
    w2 [count, H, I/2], w2_sf [count, H, I/16], w2_gs [count]. Returns bf16 [M, H]."""
    m, h = x.shape
    i_dim = w2.shape[2] * 2
    bm = bm or block_m(m)
    n_slots = m * topk
    max_blocks = triton.cdiv(n_slots, bm) + count
    dev = x.device
    sorted_ids = torch.empty((max_blocks * bm,), dtype=torch.int32, device=dev)
    block_expert = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
    _align_kernel[(1,)](topk_ids, sorted_ids, block_expert, local_start, n_slots, max_blocks,
                        COUNT=count, BINS=triton.next_power_of_2(count + 1), BM=bm,
                        NS=max(16, triton.next_power_of_2(n_slots)),
                        NB=triton.next_power_of_2(max_blocks), num_warps=4)
    inter = torch.empty((n_slots, i_dim), dtype=torch.bfloat16, device=dev)
    _gemm1_kernel[(max_blocks, i_dim // bn1)](x, sorted_ids, block_expert, w13, w13_sf, w13_gs, inter, h, i_dim,
                                              TOPK=topk, BM=bm, BN=bn1, BK=bk, num_warps=warps, num_stages=stages)
    y = torch.empty((n_slots, h), dtype=torch.bfloat16, device=dev)
    _gemm2_kernel[(max_blocks, h // bn2)](inter, sorted_ids, block_expert, w2, w2_sf, w2_gs, y, h, i_dim,
                                          BM=bm, BN=bn2, BK=bk, num_warps=warps, num_stages=stages)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    _finalize_kernel[(m,)](y, topk_ids, topk_wts, norm_w, out, local_start, h, COUNT=count, TOPK=topk,
                           BH=triton.next_power_of_2(h), EPS=eps, num_warps=8)
    return out


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def marlin_prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def marlin_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
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
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _inputs(bs, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.25 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def prepare(weights):
    t0 = time.time()
    state = marlin_prepare(weights)
    dev = weights["w13"].device
    W = {k: weights[k][:32] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__} prep {time.time() - t0:.0f}s"]

    def bench(fn, reps=20):
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

    nop = torch.zeros(1, device=dev)
    lines.append(f"nop {bench(lambda: nop.add_(1)):.1f}")
    for bs in (256, 128, 32, 16, 8, 1):
        if time.time() - t0 > 250:
            lines.append("TIME BUDGET HIT")
            break
        data = _inputs(bs, 7, dev)
        local = int(((data["topk_ids"] >= 0) & (data["topk_ids"] < 32)).sum())
        distinct = int(torch.unique(data["topk_ids"][data["topk_ids"] < 32]).numel())
        ref = marlin_fn(data, state).float()
        mt_ = bench(lambda: marlin_fn(data, state))
        res = []
        for bn1, bn2 in ((64, 64), (128, 128), (64, 128)):
            for bk in (64, 128):
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        for bm in ((16, 32, 64) if bs >= 32 else (16,)):
                            if time.time() - t0 > 250:
                                break
                            tag = f"bn{bn1}/{bn2},bk{bk},w{warps},st{stages},bm{bm}"
                            fn = lambda: moe_forward(data["hidden_states"], data["topk_ids"], data["topk_weights"],
                                                     W["w13"], W["w13_sf"], W["w13_gs"], W["w2"], W["w2_sf"], W["w2_gs"],
                                                     weights["post_norm_w"], 0, 32, bn1=bn1, bn2=bn2, bk=bk,
                                                     warps=warps, stages=stages, bm=bm)
                            try:
                                out = fn().float()
                                err = float((out - ref).norm() / ref.norm())
                                if not math.isfinite(err) or err > 0.02:
                                    res.append((1e9, tag + f"BAD{err:.1e}"))
                                    continue
                                res.append((bench(fn), tag))
                            except Exception as e:  # noqa: BLE001
                                res.append((1e9, tag + type(e).__name__[:12]))
        res.sort()
        bad = [t for v, t in res if v >= 1e9]
        lines.append(f"bs{bs} local_routes {local} distinct {distinct} marlin {mt_:.1f} | " + " ".join(f"{v:.1f}/{t}" for v, t in res[:4])
                     + f" | bad {len(bad)} {bad[0] if bad else ''} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
