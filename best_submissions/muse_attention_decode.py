# okc: problem=muse_attention_decode
"""Gated attention decode/verify over a paged FP8 cache, as three Triton kernels.

1. Prologue, one program per token: RMSNorm of q and the new keys, interleaved RoPE on sliding
   layers, the new keys and values written to their cache rows as E4M3, and the normalized query
   rounded to E4M3 (the 5.473 factor folded into the softmax scale, as FA3 does in SGLang).
2. Split-KV attention, one program per (request, KV head or head pair, key chunk): the heads' whole
   query groups (and the block's tokens) against each gathered key tile, FP8 x FP8 on the tensor
   cores for the logits, online softmax in FP32, BF16 P @ V (FP8 P @ V measured no faster: Triton's
   operand conversion eats the gain); partial max, sum and output per chunk. Decode takes both heads of a pair per program, gathering whole cache rows.
   The key loop stays branch-free: a runtime branch inside it stops Triton pipelining the loads.
3. Combine, one program per token: the chunks merged, then RMSNorm x sigmoid(gate) to BF16.

Masks are per query row: causal (a token sees keys up to its own position) and, on sliding layers,
the 2,048-token window. Lengths come from cache_seqlens on the device, so nothing depends on input
values on the host; the grid is sized from the page table's capacity.
"""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
# Aim for about this many attention programs (8 per SM), from the shapes alone.
TARGET_PROGRAMS = 1056
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
                 PV_FP8: tl.constexpr, PREFETCH: tl.constexpr):
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
    # PREFETCH: the page indices run one block ahead of the K/V loads that need them, so a block's
    # gather addresses never wait on a load issued in the same iteration (decode gains; verify,
    # which is compute-bound, measured slower with it).
    t_next = start + tl.arange(0, BLOCK_N)
    page_next = tl.load(page_table_ptr + b * max_pages + t_next // PAGE, mask=t_next < end, other=0)
    for n0 in range(start, end, BLOCK_N):
        t = n0 + tl.arange(0, BLOCK_N)
        t_ok = t < end
        if PREFETCH:
            page = page_next
            t_next = t + BLOCK_N
            page_next = tl.load(page_table_ptr + b * max_pages + t_next // PAGE, mask=t_next < end, other=0)
        else:
            page = tl.load(page_table_ptr + b * max_pages + t // PAGE, mask=t_ok, other=0)
        krow = (page.to(tl.int64) * PAGE + t % PAGE) * (HKV * D) + h0 * D
        # Rows sit at multiples of D bytes; saying so lets the gather use 16-byte vector loads
        # (the offsets come from loaded page indices, which alignment analysis can't see through).
        krow = tl.multiple_of(krow, D)
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


def _split_plan(units: int, capacity: int, block_n: int, target: int, max_splits: int):
    """(chunk, num_splits) from the shapes: about `target` programs at full capacity, at most
    `max_splits` chunks per request."""
    want = max(1, min(max_splits, target // units))
    chunk = max(block_n, triton.next_power_of_2(triton.cdiv(capacity, want)))
    return chunk, triton.cdiv(capacity, chunk)


def kernel_fn(data: Inputs) -> torch.Tensor:
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
    # Decode with a batch: both KV heads of a pair per program (whole 128-byte rows); verify's
    # 64-row tiles are compute-heavy, so there one head per program. Tile and pipeline depth per
    # layer and q_len from a sweep on the judge's H200 (probe_a1.py).
    hp = 2 if length == 1 and batch > 1 and hkv % 2 == 0 else 1
    block_n = 128 if (window == 0 and (length == 1 or batch == 1)) else 64
    stages = 3 if (window == 0 and length == 1) else 2
    units = batch * hkv // hp
    # Program target and split cap per layer and q_len from a whole-call sweep (probe_a3.py).
    target = TARGET_PROGRAMS // 2 if window and length == 1 else TARGET_PROGRAMS
    max_splits = 32 if length > 1 and batch > 1 else MAX_SPLITS
    chunk, num_splits = _split_plan(units, capacity, block_n, target, max_splits)
    rows = max(16, triton.next_power_of_2(length * group * hp))
    m = torch.empty((units, num_splits, rows), device=device, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty((units, num_splits, rows, hp * d), device=device, dtype=torch.float32)
    _attn_kernel[(units, num_splits)](
        q8, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["cache_seqlens"],
        m, l, acc, max_pages, num_splits, chunk, cfg["sm_scale"] * cfg["qk_scale_factor"],
        L=length, G=group, HP=hp, M=rows, HKV=hkv, D=d, PAGE=page, WINDOW=window, BLOCK_N=block_n,
        PV_FP8=False, PREFETCH=length == 1, num_warps=4, num_stages=stages)

    out = torch.empty((batch, length, hq * d), device=device, dtype=torch.bfloat16)
    _combine_kernel[(tokens, hq)](m, l, acc, qkv, out, num_splits, L=length, G=group, HP=hp, M=rows, HQ=hq, HKV=hkv, D=d,
                                  SB=32, EPS=RMS_EPS, num_warps=4)
    return out
