# okc: problem=muse_attention_decode
"""Tuning probe A1, not a submission: sweeps the attention kernel's BLOCK_N, warps, stages, P @ V dtype
and head pairing on the scored shapes' key cases (inputs built here like the judge's: page 1, ~80K
contexts +-7.5%, scattered pages), against SGLang's own path, as CUDA-graph replays from a cold L2.
There's no prepare() on this problem: the sweep runs on the first kernel_fn call, then stops the run."""

import math
import time
from typing import TypedDict

import triton
import triton.language as tl
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
# Aim for about this many attention programs (8 per SM), from the shapes alone.
TARGET_PROGRAMS = 1056
BLOCK_N = 64
# At most this many chunks per request: past it the combine's cost outgrows the parallelism.
MAX_SPLITS = 256
_INV_FREQ = {}


def _inv_freq(theta: float, d: int, device) -> torch.Tensor:
    """theta^(-2i/d), computed exactly as the reference does (FP32)."""
    key = (theta, d, str(device))
    if key not in _INV_FREQ:
        _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
    return _INV_FREQ[key]


@triton.jit
def _prologue_kernel(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
                     q8_ptr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
                     EPS: tl.constexpr):
    tok = tl.program_id(0)
    row = qkv_ptr + tok * ((2 * HQ + 2 * HKV) * D)
    half = tl.arange(0, D // 2)
    pos = tl.load(pos_ptr + tok).to(tl.float32)
    if ROPE:
        ang = pos * tl.load(inv_freq_ptr + half)  # FP32, as the reference: pos.float() * inv_freq
        cos = tl.cos(ang)[None, :]
        sin = tl.sin(ang)[None, :]

    # Queries: [HQ, D/2] even and odd halves of each head.
    hq = tl.arange(0, HQ)[:, None]
    qe = tl.load(row + hq * D + 2 * half[None, :]).to(tl.float32)
    qo = tl.load(row + hq * D + 2 * half[None, :] + 1).to(tl.float32)
    inv = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D + EPS)[:, None]
    qe, qo = qe * inv, qo * inv
    if ROPE:
        qe, qo = qe * cos - qo * sin, qe * sin + qo * cos
    # E4M3 rounds the normalized query before the 5.473 factor (folded into the softmax scale).
    qdst = q8_ptr + tok * (HQ * D) + hq * D + 2 * half[None, :]
    tl.store(qdst, qe.to(tl.float8e4nv))
    tl.store(qdst + 1, qo.to(tl.float8e4nv))

    # New keys: normalized (and rotated) like the queries; values as they come.
    hk = tl.arange(0, HKV)[:, None]
    kbase = row + HQ * D
    ke = tl.load(kbase + hk * D + 2 * half[None, :]).to(tl.float32)
    ko = tl.load(kbase + hk * D + 2 * half[None, :] + 1).to(tl.float32)
    kinv = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / D + EPS)[:, None]
    ke, ko = ke * kinv, ko * kinv
    if ROPE:
        ke, ko = ke * cos - ko * sin, ke * sin + ko * cos
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    loc = tl.load(loc_ptr + tok)
    kdst = k_cache_ptr + loc * (HKV * D) + hk * D + 2 * half[None, :]
    tl.store(kdst, (ke / k_scale).to(tl.float8e4nv))
    tl.store(kdst + 1, (ko / k_scale).to(tl.float8e4nv))
    d = tl.arange(0, D)[None, :]
    v = tl.load(row + (HQ + HKV) * D + hk * D + d).to(tl.float32)
    tl.store(v_cache_ptr + loc * (HKV * D) + hk * D + d, (v / v_scale).to(tl.float8e4nv))


@triton.jit
def _attn_kernel(q8_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, page_table_ptr, seqlens_ptr,
                 m_ptr, l_ptr, acc_ptr, max_pages, num_splits, chunk, qk_scale,
                 L: tl.constexpr, G: tl.constexpr, HP: tl.constexpr, M: tl.constexpr, HKV: tl.constexpr,
                 D: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr, BLOCK_N: tl.constexpr,
                 PV_FP8: tl.constexpr):
    """One (request, HP adjacent KV heads, key chunk). With HP = 2 a gathered cache row is the full
    128 bytes of both heads, and one block-diagonal query tile covers both: head hp's queries sit in
    columns hp * D .. hp * D + D - 1 (zeros elsewhere), so a single dot gives each row its own head's
    logits, and of P @ V each row keeps its own head's half (the combine reads that half)."""
    bh = tl.program_id(0)  # b * (HKV / HP) + head pair
    split = tl.program_id(1)
    b = bh // (HKV // HP)
    h0 = (bh % (HKV // HP)) * HP
    total = tl.load(seqlens_ptr + b)  # keys in the request, the block's own included
    seq = total - L                   # the block's first position
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(seq - WINDOW + 1, 0)
    start = lo + split * chunk
    end = tl.minimum(start + chunk, total)

    # Query rows: token j x head-in-pair hp x group head g, row = (j * HP + hp) * G + g.
    r = tl.arange(0, M)
    j = r // (G * HP)
    hp = (r // G) % HP
    g = r % G
    row_ok = r < L * G * HP
    c = tl.arange(0, HP * D)  # columns: head hp's D values at hp * D
    q = tl.load(q8_ptr + ((b * L + j) * (HKV * G) + (h0 + hp) * G + g)[:, None] * D + (c % D)[None, :],
                mask=row_ok[:, None] & ((c // D)[None, :] == hp[:, None]), other=0.0)
    qpos = seq + j  # each row's position

    m_i = tl.full([M], -float("inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, HP * D], tl.float32)
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    scale = qk_scale * k_scale * 1.4426950408889634
    for n0 in range(start, end, BLOCK_N):
        t = n0 + tl.arange(0, BLOCK_N)
        t_ok = t < end
        page = tl.load(page_table_ptr + b * max_pages + t // PAGE, mask=t_ok, other=0)
        krow = (page.to(tl.int64) * PAGE + t % PAGE) * (HKV * D) + h0 * D
        kt = tl.load(k_cache_ptr + krow[:, None] + c[None, :], mask=t_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kt)) * scale  # [M, BLOCK_N], log2 units
        vis = t_ok[None, :] & (t[None, :] <= qpos[:, None])
        if WINDOW > 0:
            vis = vis & (t[None, :] > qpos[:, None] - WINDOW)
        s = tl.where(vis, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vt = tl.load(v_cache_ptr + krow[:, None] + c[None, :], mask=t_ok[:, None], other=0.0)
        if PV_FP8:  # FP8 P @ V on the tensor cores, as FA3 does with an FP8 cache
            pv = tl.dot(p.to(tl.float8e4nv), vt)
        else:
            pv = tl.dot(p.to(tl.bfloat16), vt.to(tl.bfloat16))
        acc = acc * alpha[:, None] + pv
        m_i = m_new
    # Partials, [B * HKV / HP, num_splits, M] (+ HP * D for acc); empty chunks leave m = -inf, l = 0.
    base = bh * num_splits + split
    tl.store(m_ptr + base * M + r, m_i)
    tl.store(l_ptr + base * M + r, l_i)
    tl.store(acc_ptr + (base * M + r)[:, None] * (HP * D) + c[None, :], acc * v_scale)


@triton.jit
def _combine_kernel(m_ptr, l_ptr, acc_ptr, qkv_ptr, out_ptr, num_splits,
                    L: tl.constexpr, G: tl.constexpr, HP: tl.constexpr, M: tl.constexpr, HQ: tl.constexpr,
                    HKV: tl.constexpr, D: tl.constexpr, SB: tl.constexpr, EPS: tl.constexpr):
    """One (token, query head): the chunks merged SB at a time (running max), then the gated norm."""
    tok = tl.program_id(0)  # b * L + j
    hq = tl.program_id(1)
    b = tok // L
    j = tok % L
    h = hq // G
    rr = (j * HP + h % HP) * G + hq % G
    d = tl.arange(0, D)
    mx = -float("inf")
    den = 0.0
    num = tl.zeros([D], tl.float32)
    for s0 in range(0, num_splits, SB):
        s = s0 + tl.arange(0, SB)
        s_ok = s < num_splits
        base = (b * (HKV // HP) + h // HP) * num_splits + s
        m = tl.load(m_ptr + base * M + rr, mask=s_ok, other=-float("inf"))
        l = tl.load(l_ptr + base * M + rr, mask=s_ok, other=0.0)
        acc = tl.load(acc_ptr + (base * M + rr)[:, None] * (HP * D) + (h % HP) * D + d[None, :], mask=s_ok[:, None],
                      other=0.0)
        mx_new = tl.maximum(mx, tl.max(m, 0))
        safe = tl.where(mx_new == -float("inf"), 0.0, mx_new)
        w = tl.where(m == -float("inf"), 0.0, tl.exp2(m - safe))
        r = tl.where(mx == -float("inf"), 0.0, tl.exp2(mx - safe))
        den = den * r + tl.sum(l * w, 0)
        num = num * r + tl.sum(acc * w[:, None], 0)
        mx = mx_new
    o = num / den
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + EPS)
    gate = tl.load(qkv_ptr + tok * ((2 * HQ + 2 * HKV) * D) + (HQ + 2 * HKV) * D + hq * D + d).to(tl.float32)
    o = o * tl.sigmoid(gate)
    tl.store(out_ptr + tok * (HQ * D) + hq * D + d, o.to(tl.bfloat16))


def _split_plan(units: int, capacity: int):
    """(chunk, num_splits) from the shapes: about TARGET_PROGRAMS programs at full capacity."""
    want = max(1, min(MAX_SPLITS, TARGET_PROGRAMS // units))
    chunk = max(BLOCK_N, triton.next_power_of_2(triton.cdiv(capacity, want)))
    return chunk, triton.cdiv(capacity, chunk)


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def sglang_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)

    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


def _make(batch, ctx, layer, q_len, seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    hq, hkv, d = 16, 2, 64
    u = torch.rand((batch,), generator=g, device=dev)
    seq = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1)
    longest = math.ceil(ctx * 1.075)
    max_pages = longest + q_len
    total = batch * max_pages
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=dev)
    for b in range(batch):
        n = int(seq[b]) + q_len
        table[b, :n] = perm[b * max_pages: b * max_pages + n]
    kc = torch.randn((total, 1, hkv, d), generator=g, device=dev).to(torch.float8_e4m3fn)
    vc = torch.randn((total, 1, hkv, d), generator=g, device=dev).to(torch.float8_e4m3fn)
    qkv = torch.randn((batch, q_len, (2 * hq + 2 * hkv) * d), generator=g, device=dev).to(torch.bfloat16)
    pos = (seq[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    loc = torch.gather(table.long(), 1, pos.view(batch, q_len)).reshape(-1)
    cs = (seq + q_len).to(torch.int32)
    cu = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=dev)
    sched = None
    if layer == "full" and q_len == 1:
        from sgl_kernel.flash_attn import get_scheduler_metadata
        sched = get_scheduler_metadata(batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cs.max()), num_heads=hq,
                                       num_heads_k=hkv, headdim=d, cache_seqlens=cs, qkv_dtype=torch.float8_e4m3fn,
                                       cu_seqlens_q=cu, page_size=1, causal=True, num_splits=0)
    one = torch.tensor(1.0, device=dev)
    sliding = layer == "sliding"
    return {"qkv": qkv, "positions": pos, "cache_loc": loc, "cache_seqlens": cs, "cu_seqlens_q": cu,
            "scheduler_metadata": sched, "k_cache": kc, "v_cache": vc, "k_scale": one, "v_scale": one.clone(),
            "page_table": table, "seq_lens": seq.to(torch.int32),
            "config": {"batch": batch, "q_len": q_len, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d,
                       "page_size": 1, "window": 2048 if sliding else 0, "rope_theta": 500000.0 if sliding else 0.0,
                       "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0}}


def _run(data, block_n, warps, stages, pv_fp8, hp_pair, max_splits):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    window = cfg["window"]
    group = hq // hkv
    tokens = batch * length
    qkv = data["qkv"]
    device = qkv.device
    rope = bool(cfg["rope_theta"])
    inv_freq = _inv_freq(cfg["rope_theta"] or 1.0, d, device)
    q8 = torch.empty((tokens, hq, d), device=device, dtype=torch.float8_e4m3fn)
    _prologue_kernel[(tokens,)](qkv, data["positions"], data["cache_loc"], inv_freq, data["k_cache"],
                                data["v_cache"], data["k_scale"], data["v_scale"], q8,
                                HQ=hq, HKV=hkv, D=d, ROPE=rope, EPS=RMS_EPS, num_warps=4)
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    capacity = max_pages * page
    if window:
        capacity = min(capacity, window + length - 1)
    hp = 2 if hp_pair and hkv % 2 == 0 else 1
    units = batch * hkv // hp
    want = max(1, min(max_splits, TARGET_PROGRAMS // units))
    chunk = max(block_n, triton.next_power_of_2(triton.cdiv(capacity, want)))
    num_splits = triton.cdiv(capacity, chunk)
    rows = max(16, triton.next_power_of_2(length * group * hp))
    m = torch.empty((units, num_splits, rows), device=device, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty((units, num_splits, rows, hp * d), device=device, dtype=torch.float32)
    _attn_kernel[(units, num_splits)](
        q8, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["cache_seqlens"],
        m, l, acc, max_pages, num_splits, chunk, cfg["sm_scale"] * cfg["qk_scale_factor"],
        L=length, G=group, HP=hp, M=rows, HKV=hkv, D=d, PAGE=page, WINDOW=window, BLOCK_N=block_n,
        PV_FP8=pv_fp8, num_warps=warps, num_stages=stages)
    out = torch.empty((batch, length, hq * d), device=device, dtype=torch.bfloat16)
    _combine_kernel[(tokens, hq)](m, l, acc, qkv, out, num_splits, L=length, G=group, HP=hp, M=rows, HQ=hq, HKV=hkv,
                                  D=d, SB=32, EPS=RMS_EPS, num_warps=4)
    return out


_DONE = False


def kernel_fn(data):
    global _DONE
    if _DONE:
        raise RuntimeError("probe")
    _DONE = True
    t0 = time.time()
    dev = data["qkv"].device
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]

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
    shapes = [(32, "full", 8), (32, "full", 1), (1, "sliding", 1), (32, "sliding", 8), (1, "full", 1), (1, "full", 8)]
    for batch, layer, q_len in shapes:
        if time.time() - t0 > 600:
            lines.append("TIME BUDGET HIT")
            break
        data_s = _make(batch, 80000, layer, q_len)
        sg = bench(lambda: sglang_fn(data_s))
        ref = sglang_fn(data_s).float()
        res = []
        pvs = (False, True) if q_len > 1 else (True, False)
        for block_n in (32, 64, 128):
            for warps in (4, 8):
                for stages in (2, 3):
                    for pv in pvs:
                        for hp_pair in ((False,) if q_len > 1 else (True, False)):
                            for ms in ((256,) if batch > 1 else (64, 256)):
                                if time.time() - t0 > 600:
                                    break
                                tag = f"bn{block_n},w{warps},st{stages},{'fp8' if pv else 'bf16'}{',pair' if hp_pair else ''},ms{ms}"
                                try:
                                    fn = lambda: _run(data_s, block_n, warps, stages, pv, hp_pair, ms)
                                    out = fn().float()
                                    err = float((out - ref).norm() / ref.norm())
                                    if not math.isfinite(err) or err > 0.1:
                                        res.append((1e9, tag + f"BAD{err:.1e}"))
                                        continue
                                    res.append((bench(fn), tag))
                                except Exception as e:  # noqa: BLE001
                                    res.append((1e9, tag + type(e).__name__[:12]))
        res.sort()
        bad = [t for v, t in res if v >= 1e9]
        lines.append(f"B{batch} {layer} L{q_len} sglang {sg:.1f} | " + " ".join(f"{v:.1f}/{t}" for v, t in res[:5])
                     + f" | bad {len(bad)} {bad[0] if bad else ''} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
        del data_s
        torch.cuda.empty_cache()
    raise RuntimeError("PROBE\n" + "\n".join(lines))
