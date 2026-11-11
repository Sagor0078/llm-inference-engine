"""NVFP4 MoE (W4A16) as Triton grouped GEMMs: align, GEMM1 (+ SiLU x up), GEMM2, post-norm weighted sum.

Weights are repacked once (repack(), lossless nibble permutation): within each BK-wide K chunk, byte j's low
nibble holds K = j and its high nibble K = BK/2 + j, so the two dequantized planes are the chunk's two
contiguous halves and the activations load contiguously. Block scales load once per 16 values and broadcast.
With TILED (tile_major()), weights and scales are also stored tile-major, [E, N/BN, K/BK, BN, BK/2] and
[E, N/BN, K/BK, BN, BK/16]: every K-step of a tile reads one contiguous block instead of BN short row segments.

FAST dequant (inline PTX, 4 packed bytes per call): prmt spreads two bytes into the halves of a register; mask
and shift put each E2M1 code's sign at bit 15 and its 3 magnitude bits at bits 8..6, which as BF16 is exactly
value x 2^-126 (0.5 lands on a BF16 subnormal); fma.rn.bf16x2 by 2^126 (exact; BF16 FMA keeps subnormals) and
then by the E4M3 block scale (exact: <= 6 significant bits). Two values per instruction, as Marlin does.

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


_ASM_LO = """
{
.reg .b32 t, a, b, z, bias, nz;
mov.b32 z, 0;
mov.b32 bias, 0x7E807E80;
mov.b32 nz, 0x80008000;
prmt.b32 t, $2, z, 0x4140;
and.b32 a, t, 0x00070007;
shl.b32 a, a, 6;
and.b32 b, t, 0x00080008;
shl.b32 b, b, 12;
or.b32 a, a, b;
fma.rn.bf16x2 a, a, bias, nz;
fma.rn.bf16x2 $0, a, $3, nz;
prmt.b32 t, $2, z, 0x4342;
and.b32 a, t, 0x00070007;
shl.b32 a, a, 6;
and.b32 b, t, 0x00080008;
shl.b32 b, b, 12;
or.b32 a, a, b;
fma.rn.bf16x2 a, a, bias, nz;
fma.rn.bf16x2 $1, a, $4, nz;
}
"""
_ASM_HI = tl.constexpr(_ASM_LO.replace("0x00070007", "0x00700070").replace("shl.b32 a, a, 6;", "shl.b32 a, a, 2;")
                       .replace("0x00080008", "0x00800080").replace("shl.b32 b, b, 12;", "shl.b32 b, b, 8;"))
_ASM_LO = tl.constexpr(_ASM_LO)  # kernels may only read constexpr globals


@triton.jit
def _halves_scales(s_rows, kk, j16, BN: tl.constexpr, BK: tl.constexpr):
    """The bf16 block scales of K = kk .. kk + BK/2 - 1, loaded once per 16 values, broadcast to [BN, BK/2]."""
    sf = tl.load(s_rows + kk // 16 + j16[None, :]).to(tl.bfloat16)  # [BN, BK/32]
    return tl.reshape(tl.broadcast_to(sf[:, :, None], [BN, BK // 32, 16]), [BN, BK // 2])


@triton.jit
def _fp4_planes(p, sf_lo, sf_hi, FAST: tl.constexpr):
    """Packed E2M1 bytes p (uint8) and the scales of each plane (bf16) -> (low-nibble values, high-nibble values),
    both bf16 and exact."""
    if FAST:
        lo = tl.inline_asm_elementwise(_ASM_LO, "=r,=r,r,r,r", [p, sf_lo], dtype=tl.bfloat16, is_pure=True, pack=4)
        hi = tl.inline_asm_elementwise(_ASM_HI, "=r,=r,r,r,r", [p, sf_hi], dtype=tl.bfloat16, is_pure=True, pack=4)
    else:
        q = p.to(tl.int32)
        lo = _e2m1_scaled(q & 15, sf_lo.to(tl.float32))
        hi = _e2m1_scaled(q >> 4, sf_hi.to(tl.float32))
    return lo, hi


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
def _wtile(w_ptr, sf_ptr, row0, nrows: tl.constexpr, kdim: tl.constexpr, c, i, j16, BN: tl.constexpr,
           BK: tl.constexpr, TILED: tl.constexpr, FAST: tl.constexpr):
    """Rows row0 .. row0 + BN - 1 (of one expert's [nrows, kdim] matrix, the pointers already at that expert), K-chunk
    c: the two dequantized planes (bf16 [BN, BK/2] each: K = c*BK + j and c*BK + BK/2 + j)."""
    r = tl.arange(0, BN)
    if TILED:
        tile = (row0 // BN) * (kdim // BK) + c
        wp = w_ptr + tile.to(tl.int64) * (BN * (BK // 2)) + r[:, None] * (BK // 2) + i[None, :]
        sp = sf_ptr + tile.to(tl.int64) * (BN * (BK // 16)) + r[:, None] * (BK // 16)
    else:
        wp = w_ptr + (row0 + r)[:, None].to(tl.int64) * (kdim // 2) + c * (BK // 2) + i[None, :]
        sp = sf_ptr + (row0 + r)[:, None].to(tl.int64) * (kdim // 16) + c * (BK // 16)
    p = tl.load(wp)
    s_lo = tl.load(sp + j16[None, :]).to(tl.bfloat16)               # [BN, BK/32]: first half's blocks
    s_hi = tl.load(sp + BK // 32 + j16[None, :]).to(tl.bfloat16)    # second half's
    s_lo = tl.reshape(tl.broadcast_to(s_lo[:, :, None], [BN, BK // 32, 16]), [BN, BK // 2])
    s_hi = tl.reshape(tl.broadcast_to(s_hi[:, :, None], [BN, BK // 32, 16]), [BN, BK // 2])
    return _fp4_planes(p, s_lo, s_hi, FAST)


@triton.jit(do_not_specialize=["SPLIT"])
def _gemm1_kernel(x_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, inter_ptr, wsg_ptr, wsu_ptr, cnt_ptr,
                  SPLIT, H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, FAST: tl.constexpr, TILED: tl.constexpr):
    """inter[slot, n] = silu(x[tok] @ gate[e, n]^T) * (x[tok] @ up[e, n]^T) for BM slots x BN columns, over this
    unit's 1/SPLIT of K. With SPLIT > 1 the partials meet in zeroed FP32 workspaces and the last unit to finish a
    tile applies SiLU x up and re-zeroes them (the split-K hand-off proven on muse_bf16_proj)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    tok = tl.where(row_ok, slot // TOPK, 0)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    j16 = tl.arange(0, BK // 32)
    e64 = e.to(tl.int64)
    we = w_ptr + e64 * (2 * I) * (H // 2)       # this expert's [2I, H/2] codes (gate rows, then up rows)
    se = sf_ptr + e64 * (2 * I) * (H // 16)
    xrow = x_ptr + tok[:, None].to(tl.int64) * H
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    chunks = H // BK
    c0 = pid_k * chunks // SPLIT
    c1 = (pid_k + 1) * chunks // SPLIT
    for c in range(c0, c1):
        k0 = c * BK
        xa = tl.load(xrow + k0 + i[None, :], mask=row_ok[:, None], other=0.0)
        xb = tl.load(xrow + k0 + BK // 2 + i[None, :], mask=row_ok[:, None], other=0.0)
        g_lo, g_hi = _wtile(we, se, pid_n * BN, 2 * I, H, c, i, j16, BN, BK, TILED, FAST)
        u_lo, u_hi = _wtile(we, se, I + pid_n * BN, 2 * I, H, c, i, j16, BN, BK, TILED, FAST)
        acc_g = tl.dot(xa, tl.trans(g_lo), acc_g)
        acc_g = tl.dot(xb, tl.trans(g_hi), acc_g)
        acc_u = tl.dot(xa, tl.trans(u_lo), acc_u)
        acc_u = tl.dot(xb, tl.trans(u_hi), acc_u)
    srow = tl.where(row_ok, slot, 0)[:, None].to(tl.int64)
    if SPLIT > 1:
        wsg = wsg_ptr + srow * I + n[None, :]
        wsu = wsu_ptr + srow * I + n[None, :]
        tl.atomic_add(wsg, acc_g, mask=row_ok[:, None], sem="relaxed")
        tl.atomic_add(wsu, acc_u, mask=row_ok[:, None], sem="relaxed")
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(1) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT - 1:
            acc_g = tl.load(wsg, mask=row_ok[:, None], other=0.0, cache_modifier=".cg")
            acc_u = tl.load(wsu, mask=row_ok[:, None], other=0.0, cache_modifier=".cg")
            gs = tl.load(gs_ptr + e)
            g = acc_g * gs
            h = g * tl.sigmoid(g) * (acc_u * gs)
            tl.store(inter_ptr + srow * I + n[None, :], h.to(tl.bfloat16), mask=row_ok[:, None])
            tl.debug_barrier()  # every thread has read before any re-zeroes
            tl.store(wsg, tl.zeros([BM, BN], tl.float32), mask=row_ok[:, None])
            tl.store(wsu, tl.zeros([BM, BN], tl.float32), mask=row_ok[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")
    else:
        gs = tl.load(gs_ptr + e)
        g = acc_g * gs
        h = g * tl.sigmoid(g) * (acc_u * gs)
        tl.store(inter_ptr + srow * I + n[None, :], h.to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit(do_not_specialize=["SPLIT"])
def _gemm2_kernel(inter_ptr, sorted_ptr, block_expert_ptr, w_ptr, sf_ptr, gs_ptr, y_ptr, wsy_ptr, cnt_ptr,
                  SPLIT, H: tl.constexpr, I: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FAST: tl.constexpr, TILED: tl.constexpr):
    """y[slot, n] = inter[slot] @ w2[e, n]^T for BM slots x BN columns, split-K as GEMM1."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    r = tl.arange(0, BM)
    slot = tl.load(sorted_ptr + pid_m * BM + r)
    row_ok = slot >= 0
    srow = tl.where(row_ok, slot, 0).to(tl.int64)
    n = pid_n * BN + tl.arange(0, BN)
    i = tl.arange(0, BK // 2)
    j16 = tl.arange(0, BK // 32)
    e64 = e.to(tl.int64)
    we = w_ptr + e64 * H * (I // 2)
    se = sf_ptr + e64 * H * (I // 16)
    arow = inter_ptr + srow[:, None] * I
    acc = tl.zeros([BM, BN], tl.float32)
    chunks = I // BK
    c0 = pid_k * chunks // SPLIT
    c1 = (pid_k + 1) * chunks // SPLIT
    for c in range(c0, c1):
        k0 = c * BK
        aa = tl.load(arow + k0 + i[None, :], mask=row_ok[:, None], other=0.0)
        ab = tl.load(arow + k0 + BK // 2 + i[None, :], mask=row_ok[:, None], other=0.0)
        w_lo, w_hi = _wtile(we, se, pid_n * BN, H, I, c, i, j16, BN, BK, TILED, FAST)
        acc = tl.dot(aa, tl.trans(w_lo), acc)
        acc = tl.dot(ab, tl.trans(w_hi), acc)
    dst = y_ptr + srow[:, None] * H + n[None, :]
    if SPLIT > 1:
        ws = wsy_ptr + srow[:, None] * H + n[None, :]
        tl.atomic_add(ws, acc, mask=row_ok[:, None], sem="relaxed")
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(1) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT - 1:
            total = tl.load(ws, mask=row_ok[:, None], other=0.0, cache_modifier=".cg")
            tl.store(dst, (total * tl.load(gs_ptr + e)).to(tl.bfloat16), mask=row_ok[:, None])
            tl.debug_barrier()  # every thread has read before any re-zeroes
            tl.store(ws, tl.zeros([BM, BN], tl.float32), mask=row_ok[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")
    else:
        tl.store(dst, (acc * tl.load(gs_ptr + e)).to(tl.bfloat16), mask=row_ok[:, None])


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


def repack(packed: torch.Tensor, bk: int) -> torch.Tensor:
    """[..., K/2] uint8 (byte i = K 2i low nibble, 2i+1 high) -> the same codes with, in each bk-wide K chunk, byte j
    = K j (low nibble) and K bk/2 + j (high nibble). A permutation: lossless."""
    lo, hi = packed & 15, packed >> 4
    codes = torch.stack((lo, hi), -1).flatten(-2)  # [..., K]
    k = codes.shape[-1]
    c = codes.view(*codes.shape[:-1], k // bk, 2, bk // 2)
    return (c[..., 0, :] | (c[..., 1, :] << 4)).reshape(*codes.shape[:-1], k // 2).contiguous()


def tile_major(t: torch.Tensor, bn: int, bk: int, per: int) -> torch.Tensor:
    """[E, N, K/per] -> [E, N/bn, K/bk, bn, bk/per]: the same bytes, tile-major (lossless). per = 2 for packed
    codes (after repack(..., bk)), 16 for block scales."""
    e, n, kp = t.shape
    return t.view(e, n // bn, bn, kp // (bk // per), bk // per).permute(0, 1, 3, 2, 4).contiguous()


def block_m(m: int) -> int:
    """Rows per expert block from the token count (shape-only)."""
    return 16 if m <= 16 else 32 if m <= 64 else 64


def make_workspace(max_slots: int, h: int, i_dim: int, device) -> dict:
    """Zeroed split-K workspaces (re-zeroed by the kernels after every use) for up to max_slots routed slots."""
    return {"g": torch.zeros((max_slots * i_dim,), dtype=torch.float32, device=device),
            "u": torch.zeros((max_slots * i_dim,), dtype=torch.float32, device=device),
            "y": torch.zeros((max_slots * h,), dtype=torch.float32, device=device),
            "cnt1": torch.zeros((1 << 16,), dtype=torch.int32, device=device),
            "cnt2": torch.zeros((1 << 16,), dtype=torch.int32, device=device), "max_slots": max_slots}


def moe_forward(x, topk_ids, topk_wts, w13, w13_sf, w13_gs, w2, w2_sf, w2_gs, norm_w, local_start: int,
                count: int, topk: int = 8, bn1: int = 64, bn2: int = 64, bk: int = 64, warps: int = 4,
                stages: int = 3, eps: float = 1e-8, bm: int = 0, fast: bool = True, split1: int = 1, split2: int = 1,
                ws=None, tiled: bool = False):
    """w13 / w2 repacked with repack(..., bk). split1 / split2 > 1 need ws (make_workspace) covering m * topk slots.
    Returns bf16 [M, H]."""
    m, h = x.shape
    i_dim = w2.shape[-1] * w2.shape[-3] * 2 if tiled else w2.shape[2] * 2
    bm = bm or block_m(m)
    n_slots = m * topk
    max_blocks = triton.cdiv(n_slots, bm) + count
    dev = x.device
    if split1 > 1 or split2 > 1:
        assert ws is not None and n_slots <= ws["max_slots"]
        assert max_blocks * (max(i_dim // bn1, h // bn2)) <= ws["cnt1"].numel()
    dummy = torch.empty((1,), dtype=torch.float32, device=dev)
    sorted_ids = torch.empty((max_blocks * bm,), dtype=torch.int32, device=dev)
    block_expert = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
    _align_kernel[(1,)](topk_ids, sorted_ids, block_expert, local_start, n_slots, max_blocks,
                        COUNT=count, BINS=triton.next_power_of_2(count + 1), BM=bm,
                        NS=max(16, triton.next_power_of_2(n_slots)), NB=triton.next_power_of_2(max_blocks), num_warps=4)
    inter = torch.empty((n_slots, i_dim), dtype=torch.bfloat16, device=dev)
    _gemm1_kernel[(max_blocks, i_dim // bn1, split1)](
        x, sorted_ids, block_expert, w13, w13_sf, w13_gs, inter, ws["g"] if ws else dummy, ws["u"] if ws else dummy,
        ws["cnt1"] if ws else dummy, split1, H=h, I=i_dim, TOPK=topk, BM=bm, BN=bn1, BK=bk, FAST=fast, TILED=tiled, num_warps=warps,
        num_stages=stages)
    y = torch.empty((n_slots, h), dtype=torch.bfloat16, device=dev)
    _gemm2_kernel[(max_blocks, h // bn2, split2)](
        inter, sorted_ids, block_expert, w2, w2_sf, w2_gs, y, ws["y"] if ws else dummy, ws["cnt2"] if ws else dummy,
        split2, H=h, I=i_dim, BM=bm, BN=bn2, BK=bk, FAST=fast, TILED=tiled, num_warps=warps, num_stages=stages)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    _finalize_kernel[(m,)](y, topk_ids, topk_wts, norm_w, out, local_start, h, COUNT=count, TOPK=topk,
                           BH=triton.next_power_of_2(h), EPS=eps, num_warps=8)
    return out
