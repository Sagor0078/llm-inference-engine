# okc: problem=muse_bf16_proj
"""Tuning probe, not a submission: sweeps kernel configs inside prepare() on the judge's H200,
timing CUDA-graph replays from a cold L2, then stops the run with the table in the error."""

import itertools
import time

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _proj_kernel(x_ptr, w_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HI_LO: tl.constexpr, TILED: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = c0 * BK + tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    if TILED:  # [N/BN, K/BK, BN, BK]: each unit's k-range is one contiguous run
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
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(0) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            total = tl.load(ws_ptrs, mask=m_mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask)
            tl.store(ws_ptrs, tl.zeros((BM, BN), dtype=tl.float32), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")


LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "router": (256, 8192)}
SMS = 132


def tile(w, bn, bk):
    n, k = w.shape
    return w.reshape(n // bn, bn, k // bk, bk).permute(0, 2, 1, 3).contiguous()


def prepare(weights):
    t0 = time.time()
    dev = weights["router"].device
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    ws = torch.zeros(1 << 22, device=dev, dtype=torch.float32)
    cnt = torch.zeros(1 << 14, device=dev, dtype=torch.int32)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)  # 256 MB, read (as the judge does)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]
    tiles_cache = {}

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

    shapes = ([(m, l) for m in (1, 16, 32) for l in ("qkv_gate", "o", "pooler")] + [(1, "unpooler"), (8, "unpooler"), (16, "unpooler"), (32, "unpooler")]
              + [(m, "router") for m in (1, 32, 128, 256)] + [(m, l) for m in (128, 256) for l in ("qkv_gate", "o", "pooler", "unpooler")])
    for m, layer in shapes:
        if time.time() - t0 > 270:
            lines.append("TIME BUDGET HIT")
            break
        n, k = LAYERS[layer]
        if layer == "unpooler" and m <= 8:
            n = 1024
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        is_r = layer == "router"
        w = hi if is_r else weights[layer]
        ref = x.double() @ weights[layer][:n].double().T
        if is_r:
            out = torch.empty((m, n), device=dev, dtype=torch.float32)
            base = bench(lambda: torch.mm(x, hi.T, out_dtype=torch.float32).add_(torch.mm(x, lo.T, out_dtype=torch.float32))) \
                if m > 8 else bench(lambda: x.float() @ weights["router"].T)
        else:
            out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
            base = bench(lambda: x @ w[:n].T)
        if m <= 64:
            bm = max(16, triton.next_power_of_2(m))
            space = [(bm, bn, bk, 4, st) for bn in (16, 32, 64) for bk in (128, 256, 512) for st in (3, 4)
                     if bn * bk <= 16384 and not (is_r and bk == 512)]
            targets = (SMS, 2 * SMS, 4 * SMS)
        else:
            space = [(bm, bn, bk, wp, st) for bm in (64, 128) for bn in (32, 64, 128) for bk in (64, 128) for wp in (4, 8)
                     for st in (3, 4) if not (wp == 8 and bm * bn < 8192) and bm * bn <= 8192]
            targets = (1, SMS, 2 * SMS)
        res = []
        for (bm, bn, bk, warps, stages), tiled in itertools.product(space, (False, True)):
            if time.time() - t0 > 270:
                break
            if tiled:
                key = (layer, bn, bk)
                if key not in tiles_cache:
                    tiles_cache[key] = (tile(w, bn, bk), tile(lo, bn, bk) if is_r else None)
                wt, lot = tiles_cache[key]
                lot = lot if is_r else wt
            else:
                wt, lot = w, lo
            ntiles = triton.cdiv(m, bm) * (n // bn)
            chunks = k // bk
            for split in sorted({max(1, min(chunks, round(t / ntiles))) for t in targets}):
                grid = (n // bn, split, triton.cdiv(m, bm))

                def run(bm=bm, bn=bn, bk=bk, warps=warps, stages=stages, split=split, grid=grid, wt=wt, lot=lot, tiled=tiled):
                    _proj_kernel[grid](x, wt, lot, out, ws, cnt, m, n, k, split, x.stride(0), k,
                                       out.stride(0), BM=bm, BN=bn, BK=bk, HI_LO=is_r, TILED=tiled, num_warps=warps,
                                       num_stages=stages)
                try:
                    run()
                    err = float((out.double() - ref).norm() / ref.norm())
                    if err > (1e-4 if is_r else 1e-2) or cnt.abs().max() != 0:
                        res.append((1e9, f"BAD{err:.0e}", bm, bn, bk, warps, stages, split, tiled))
                        continue
                    res.append((bench(run), "", bm, bn, bk, warps, stages, split, tiled))
                except Exception as e:  # noqa: BLE001
                    res.append((1e9, type(e).__name__[:8], bm, bn, bk, warps, stages, split, tiled))
        res.sort()
        def fmt(r):
            t, bad, bm, bn, bk, wp, st, sp, tl_ = r
            return f"{t:.1f}/{'T' if tl_ else 'R'}{bm},{bn},{bk},{wp},{st},s{sp}{bad}"
        best_r = next((r for r in res if not r[8]), None)
        best_t = next((r for r in res if r[8]), None)
        top = " ".join(fmt(r) for r in res[:4])
        bad = sum(1 for r in res if r[1])
        lines.append(f"{layer[:4]} m{m} base {base:.1f} | {top} | bestR {fmt(best_r) if best_r else '-'} bestT {fmt(best_t) if best_t else '-'} | bad {bad} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("TUNE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("tuning probe")
