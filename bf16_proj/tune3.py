# okc: problem=muse_bf16_proj
"""Tuning probe 3, not a submission: swap-AB TMA kernel (weights as the wide MMA operand, all of m in
one tile) against the current kernel and cuBLAS, timed as CUDA-graph replays from a cold L2."""

import itertools
import time

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

WS_FLOATS = 1 << 20
WS_TILES = 4096

# Per BF16 layer: [(max m, (BM, BN, BK, num_warps, num_stages, SPLIT_K, TILED))], BM 0 meaning the
# smallest MMA tile that covers m; past the last entry, cuBLAS. TILED reads the tile-major copy.
BF16_CONFIGS = {
    "qkv_gate": [(32, (0, 64, 128, 4, 3, 7, True))],
    "o": [(32, (0, 32, 128, 4, 4, 1, False))],
    "pooler": [(32, (0, 64, 128, 4, 3, 4, True))],
    "unpooler_slice": [(8, (0, 64, 128, 4, 4, 8, False))],  # m <= 8: this GPU's 1024-row slice
    "unpooler": [(16, (0, 64, 256, 4, 3, 1, True))],
}
# Tile-major copies, [N/BN, K/BK, BN, BK], for the configs that read them: (BN, BK) per layer.
TILED_LAYOUTS = {"qkv_gate": (64, 128), "pooler": (64, 128), "unpooler": (64, 256)}


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _proj_kernel(x_ptr, w_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HI_LO: tl.constexpr, TILED: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    # This unit's share of the K chunks, balanced for any SPLIT_K.
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = c0 * BK + tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    if TILED:  # each unit's k-range of its tile is one contiguous run
        w_off = (pid_n * chunks + c0) * (BN * BK) + tl.arange(0, BN)[:, None] * BK + tl.arange(0, BK)[None, :]
    else:
        w_off = offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(c0, c1):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptr + w_off, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
        if HI_LO:
            lo = tl.load(lo_ptr + w_off, eviction_policy="evict_first")
            acc = tl.dot(x, tl.trans(lo), acc)
        x_ptrs += BK
        w_off += BN * BK if TILED else BK

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask)
    else:
        ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.atomic_add(ws_ptrs, acc, mask=m_mask, sem="relaxed")
        # All of this unit's adds land before it counts itself in; the last one in reduces.
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(0) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            # The others are done with this tile; read past L1 (the adds landed in L2), then re-zero.
            total = tl.load(ws_ptrs, mask=m_mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask)
            # Every thread has read before any re-zeroes: the zeros can land in another thread's
            # layout than the loads (BM 128 on 8 warps lost partial sums without this).
            tl.debug_barrier()
            tl.store(ws_ptrs, tl.zeros((BM, BN), dtype=tl.float32), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")


def _config(layer: str, m: int, n: int, k: int):
    """(BM, BN, BK, num_warps, num_stages, SPLIT_K, TILED) for a shape, or None for cuBLAS. Shape-only."""
    if n % 64 or k % 256:
        return None
    small_bm = max(16, triton.next_power_of_2(m))
    if layer == "router":
        if m <= 16:
            return (small_bm, 32, 128, 4, 4, 16, False)
        if m <= 32:
            return (small_bm, 64, 128, 4, 4, 33, False)
        if m <= 128:
            return (64, 32, 128, 4, 3, 16, False)
        tiles = triton.cdiv(m, 64) * (n // 64)
        return (64, 64, 128, 4, 4, max(1, min(8, 264 // tiles)), False)
    key = "unpooler_slice" if layer == "unpooler" and n <= 1024 else layer
    for max_m, cfg in BF16_CONFIGS.get(key, ()):
        if m <= max_m:
            return (cfg[0] or small_bm,) + cfg[1:]
    return None


def _tile(w: torch.Tensor, bn: int, bk: int) -> torch.Tensor:
    """[N, K] -> [N/BN, K/BK, BN, BK]: the same values, tile-major (lossless repack)."""
    n, k = w.shape
    return w.reshape(n // bn, bn, k // bk, bk).permute(0, 2, 1, 3).contiguous()



@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _swab_kernel(w_desc, lo_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HI_LO: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for c in range(c0, c1):
        x = x_desc.load([pid_m * BM, c * BK])
        w = w_desc.load([pid_n * BN, c * BK])
        acc = tl.dot(w, tl.trans(x), acc)
        if HI_LO:
            lo = lo_desc.load([pid_n * BN, c * BK])
            acc = tl.dot(lo, tl.trans(x), acc)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        ws_ptrs = ws_ptr + offs_m[None, :] * N + offs_n[:, None]
        tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(0) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            total = tl.load(ws_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=mask)
            tl.debug_barrier()
            tl.store(ws_ptrs, tl.zeros((BN, BM), dtype=tl.float32), mask=mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")


SMS = 132


def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    for layer, (bn, bk) in TILED_LAYOUTS.items():
        state[f"_tiled_{layer}"] = _tile(state[layer], bn, bk)
    hi = state["router"].to(torch.bfloat16)
    lo = (state["router"] - hi.float()).to(torch.bfloat16)
    dev = hi.device
    ws = torch.zeros(WS_FLOATS * 4, device=dev, dtype=torch.float32)
    cnt = torch.zeros(WS_TILES, device=dev, dtype=torch.int32)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]

    def bench(fn, reps=25):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ts = []
        for _ in range(reps):
            torch.sum(flush, dim=0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    nop = torch.zeros(1, device=dev)
    lines.append(f"nop {bench(lambda: nop.add_(1)):.1f}")
    shapes = ([(128, l) for l in ("pooler", "unpooler", "qkv_gate", "o", "router")]
              + [(256, l) for l in ("pooler", "unpooler", "qkv_gate", "o", "router")]
              + [(m, l) for m in (1, 16, 32) for l in ("pooler", "qkv_gate", "unpooler", "o", "router")])
    for m, layer in shapes:
        if time.time() - t0 > 265:
            lines.append("TIME BUDGET HIT")
            break
        n, k = weights[layer].shape
        if layer == "unpooler" and m <= 8:
            n = 1024
        is_r = layer == "router"
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), device=dev, dtype=torch.float32 if is_r else torch.bfloat16)
        ref = (x.double() @ state[layer][:n].double().T)
        if is_r:
            base = bench(lambda: torch.mm(x, hi.T, out_dtype=torch.float32).add_(torch.mm(x, lo.T, out_dtype=torch.float32))) \
                if m > 8 else bench(lambda: x.float() @ state["router"].T)
        else:
            base = bench(lambda: x @ state[layer][:n].T)
        # The current kernel's config for this shape.
        cfg = _config(layer, m, n, k)
        cur = None
        if cfg is not None:
            bm, bn, bk, warps, stages, split, tiled = cfg
            w = hi if is_r else (state[f"_tiled_{layer}"] if tiled else state[layer])
            lo_ = lo if is_r else w
            grid = (n // bn, split, triton.cdiv(m, bm))
            cur = bench(lambda: _proj_kernel[grid](x, w, lo_, out, ws, cnt, m, n, k, split, x.stride(0), k, out.stride(0),
                                                   BM=bm, BN=bn, BK=bk, HI_LO=is_r, TILED=tiled, num_warps=warps,
                                                   num_stages=stages))
        wsrc = hi if is_r else state[layer]
        if m >= 64:
            bms = (min(256, triton.next_power_of_2(m)),)
            space = [(bm, bn, bk, wp, st) for bm in bms for bn in (64, 128) for bk in (64, 128) for wp in (4, 8)
                     for st in (3, 4) if (bn * bk + bm * bk) * 2 * (2 if is_r else 1) * st <= 200 * 1024
                     and bn * bm // (wp * 32) <= 128 and not (bn == 128 and wp == 4)]
            targets = (1, SMS, 2 * SMS)
        else:
            bm = max(16, triton.next_power_of_2(m))
            space = [(bm, bn, bk, 4, st) for bn in (64, 128) for bk in (128, 256) for st in (3, 4, 6)
                     if (bn * bk + bm * bk) * 2 * (2 if is_r else 1) * st <= 200 * 1024]
            targets = (SMS, 2 * SMS, 4 * SMS)
        res = []
        for bm, bn, bk, warps, stages in space:
            if time.time() - t0 > 265:
                break
            tiles = triton.cdiv(m, bm) * (n // bn)
            chunks = k // bk
            wd = TensorDescriptor.from_tensor(wsrc[:n], [bn, bk])
            ld = TensorDescriptor.from_tensor(lo, [bn, bk]) if is_r else wd
            xd = TensorDescriptor.from_tensor(x, [bm, bk])
            for split in sorted({max(1, min(chunks, round(t / tiles))) for t in targets}):
                if split > 1 and m * n > ws.numel():
                    continue
                grid = (n // bn, split, triton.cdiv(m, bm))

                def run(bm=bm, bn=bn, bk=bk, warps=warps, stages=stages, split=split, grid=grid, wd=wd, ld=ld, xd=xd):
                    _swab_kernel[grid](wd, ld, xd, out, ws, cnt, m, n, k, split, out.stride(0),
                                       BM=bm, BN=bn, BK=bk, HI_LO=is_r, num_warps=warps, num_stages=stages)
                try:
                    run()
                    err = float((out.double() - ref).norm() / ref.norm())
                    if err > (1e-4 if is_r else 1e-2) or cnt.abs().max() != 0:
                        res.append((1e9, f"BAD{err:.0e}", bm, bn, bk, warps, stages, split))
                        continue
                    res.append((bench(run), "", bm, bn, bk, warps, stages, split))
                except Exception as e:  # noqa: BLE001
                    res.append((1e9, type(e).__name__[:8], bm, bn, bk, warps, stages, split))
        res.sort()
        top = " ".join(f"{t:.1f}/{bm},{bn},{bk},{wp},{st},s{sp}{bad}" for t, bad, bm, bn, bk, wp, st, sp in res[:4])
        bad = sum(1 for r in res if r[1])
        why = next((r[1] for r in res if r[1]), "")
        lines.append(f"{layer[:4]} m{m} base {base:.1f} cur {cur if cur is None else round(cur, 1)} | swab {top} | bad {bad} {why} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("TUNE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("tuning probe")
