# okc: problem=muse_bf16_proj
"""Probe 7, not a submission: replacements for the split-K hand-off. E0 current; E1 atomics-only
main kernel + a reduce kernel chained by programmatic dependent launch; E2 the same without PDL;
E3 (router) zeroed FP32 output with atomics straight into it. CUDA-graph replays, cold L2."""

import time

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

WS_FLOATS = 1 << 20
WS_TILES = 4096
SKINNY_MAX_M = 32

# (BN, BK, num_warps, num_stages, SPLIT_K) per BF16 layer for m <= 32, swap-AB TMA kernel.
BF16_CONFIGS = {
    "qkv_gate": (64, 128, 4, 4, 7),
    "o": (64, 256, 4, 4, 1),
    "pooler": (64, 128, 4, 3, 4),
    "unpooler_slice": (64, 128, 4, 3, 16),  # m <= 8: this GPU's 1024-row slice
    "unpooler": (64, 256, 4, 4, 1),
    "pooler_m128": (64, 128, 4, 4, 2),  # 64 < m <= 128: the one wider shape that beats cuBLAS
}



@triton.jit
def _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, tile, SPLIT_K, pid_k, MODE: tl.constexpr):
    if MODE == 0:
        if SPLIT_K == 1:
            tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=mask)
        else:
            tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
            tl.debug_barrier()
            done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
            if done == SPLIT_K - 1:
                total = tl.load(ws_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
                tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=mask)
                tl.debug_barrier()
                tl.store(ws_ptrs, tl.zeros_like(acc), mask=mask)
                tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")
    elif MODE == 1:  # atomics only (a reduce kernel follows)
        tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
    elif MODE == 2:  # plain store of this unit's partial (racy values; timing only)
        tl.store(ws_ptrs, acc, mask=mask)
    else:  # keep the loads alive, write almost nothing
        s = tl.sum(tl.sum(acc, axis=1), axis=0)
        if s == 1234.5:
            tl.store(cnt_ptr + tile, 1)


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _swab_kernel(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    gdc_launch_dependents()
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
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    ws_ptrs = ws_ptr + offs_m[None, :] * N + offs_n[:, None]
    _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K, pid_k, MODE)


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _router_kernel(x_ptr, hi_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_xm, stride_wn,
                   stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    gdc_launch_dependents()
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = c0 * BK + tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(c0, c1):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        hi = tl.load(hi_ptr + w_off, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(hi), acc)
        lo = tl.load(lo_ptr + w_off, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(lo), acc)
        x_ptrs += BK
        w_off += BK
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
    _finish(acc, out_ptrs, ws_ptrs, m_mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K, pid_k, MODE)


@triton.jit
def _reduce_kernel(ws_ptr, out_ptr, total, BLOCK: tl.constexpr):
    """out = ws (fp32 sums) cast to out's dtype; ws re-zeroed. Waits for the main kernel under PDL."""
    gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(ws_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(ws_ptr + offs, tl.zeros_like(v), mask=mask)


def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = state["router"].to(torch.bfloat16)
    lo = (state["router"] - hi.float()).to(torch.bfloat16)
    dev = hi.device
    ws = torch.zeros(1 << 22, device=dev, dtype=torch.float32)
    cnt = torch.zeros(WS_TILES, device=dev, dtype=torch.int32)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]

    def bench(fn, reps=40):
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
    lines.append(f"nop {bench(lambda: nop.add_(1)):.2f}")
    def reduce(out, pdl):
        total = out.numel()
        _reduce_kernel[(triton.cdiv(total, 1024),)](ws, out, total, BLOCK=1024, num_warps=4, launch_pdl=pdl)

    for m, layer in ((1, "unpooler"), (8, "unpooler"), (1, "router"), (16, "router"), (32, "router"),
                     (1, "qkv_gate"), (16, "qkv_gate"), (1, "pooler"), (32, "pooler")):
        n, k = state[layer].shape
        if layer == "unpooler" and m <= 8:
            n = 1024
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        is_r = layer == "router"
        out = torch.empty((m, n), device=dev, dtype=torch.float32 if is_r else torch.bfloat16)
        ref = (x.double() @ state[layer][:n].double().T)
        row = []
        if is_r:
            bm, bn, bk, warps, stages, split = (16, 32, 128, 4, 4, 16) if m <= 16 else (32, 64, 128, 4, 4, 33)
            grid = (n // bn, split, 1)

            def main(mode, target):
                _router_kernel[grid](x, hi, lo, target, target if mode == 4 else ws, cnt, m, n, k, split, x.stride(0), k,
                                     target.stride(0), BM=bm, BN=bn, BK=bk, MODE=1 if mode == 4 else mode,
                                     num_warps=warps, num_stages=stages)
        else:
            key = "unpooler_slice" if layer == "unpooler" and n <= 1024 else layer
            bn, bk, warps, stages, split = BF16_CONFIGS[key]
            bm = max(16, triton.next_power_of_2(m))
            wd = TensorDescriptor.from_tensor(state[layer][:n], [bn, bk])
            xd = TensorDescriptor.from_tensor(x, [bm, bk])
            grid = (n // bn, split, 1)

            def main(mode, target):
                _swab_kernel[grid](wd, xd, target, ws, cnt, m, n, k, split, target.stride(0),
                                   BM=bm, BN=bn, BK=bk, MODE=mode, num_warps=warps, num_stages=stages)
        variants = {"E0": lambda: main(0, out),
                    "E1pdl": lambda: (main(1, out), reduce(out, True)),
                    "E2plain": lambda: (main(1, out), reduce(out, False))}
        if is_r:
            variants["E3zero"] = lambda: (out.zero_(), main(4, out))
        for name, fn in variants.items():
            try:
                fn()
                torch.cuda.synchronize()
                err = float((out.double() - ref).norm() / ref.norm())
                ok = err < (1e-4 if is_r else 1e-2) and float(ws.abs().max()) == 0.0 and int(cnt.abs().max()) == 0
                t = bench(fn)
                # Correct after many replays too (the workspace must come back zeroed every time).
                fn()
                torch.cuda.synchronize()
                err2 = float((out.double() - ref).norm() / ref.norm())
                ok = ok and err2 < (1e-4 if is_r else 1e-2)
                row.append(f"{name}={t:.2f}{'' if ok else f'!BAD{err:.0e}/{err2:.0e}'}")
            except Exception as e:  # noqa: BLE001
                row.append(f"{name}=EXC:{type(e).__name__}:{str(e)[:60]}")
            ws.zero_(); cnt.zero_()
        lines.append(f"{layer[:4]} m{m} n{n} split{split} | " + " ".join(row) + f" [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
