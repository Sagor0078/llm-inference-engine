"""NVFP4 MoE (W4A16) as Triton grouped GEMMs: align, GEMM1 (+ SiLU x up), GEMM2, post-norm weighted sum.

Weights stay NVFP4. A tile is dequantized in registers: E2M1 code -> value, times its E4M3 block
scale, exactly representable in BF16 (<= 6 significant bits); the expert's FP32 global scale is
applied to the FP32 accumulator, so the dot runs on exact weight values. Each packed byte holds K
elements 2i (low nibble) and 2i+1 (high nibble): the two nibble planes are dotted against the even
and odd activation columns, so nothing is interleaved.
"""

import torch
import triton
import triton.language as tl


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
