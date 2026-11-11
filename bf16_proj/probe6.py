# okc: problem=muse_bf16_proj
"""Probe 6, not a submission: where the small shapes' fixed time goes. The swap-AB and router
kernels with their split-K epilogue cut down step by step (full protocol, atomics only, plain
partial stores, no write), timed as CUDA-graph replays from a cold L2."""

import time

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

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
    elif MODE == 1:  # atomics only
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
    for m, layer in ((1, "unpooler"), (8, "unpooler"), (1, "router"), (16, "router"), (1, "o"), (32, "o"),
                     (1, "qkv_gate"), (1, "pooler"), (16, "unpooler")):
        n, k = state[layer].shape
        if layer == "unpooler" and m <= 8:
            n = 1024
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        is_r = layer == "router"
        out = torch.empty((m, n), device=dev, dtype=torch.float32 if is_r else torch.bfloat16)
        row = []
        if is_r:
            for (bm, bn, bk, warps, stages, split) in ((16 if m <= 16 else 32, 32, 128, 4, 4, 16), (16, 32, 128, 4, 4, 8), (16, 64, 128, 4, 4, 4)):
                for mode in (0, 1, 2, 3):
                    grid = (n // bn, split, 1)
                    t = bench(lambda: _router_kernel[grid](x, hi, lo, out, ws, cnt, m, n, k, split, x.stride(0), k, out.stride(0),
                                                           BM=bm, BN=bn, BK=bk, MODE=mode, num_warps=warps, num_stages=stages))
                    ws.zero_(); cnt.zero_()
                    row.append(f"bn{bn}s{split}M{mode}={t:.2f}")
        else:
            key = "unpooler_slice" if layer == "unpooler" and n <= 1024 else layer
            bn, bk, warps, stages, split = BF16_CONFIGS[key]
            bm = max(16, triton.next_power_of_2(m))
            for sp in sorted({split, max(1, split // 2), 1}):
                wd = TensorDescriptor.from_tensor(state[layer][:n], [bn, bk])
                xd = TensorDescriptor.from_tensor(x, [bm, bk])
                for mode in (0, 1, 2, 3):
                    grid = (n // bn, sp, 1)
                    t = bench(lambda: _swab_kernel[grid](wd, xd, out, ws, cnt, m, n, k, sp, out.stride(0),
                                                         BM=bm, BN=bn, BK=bk, MODE=mode, num_warps=warps, num_stages=stages))
                    ws.zero_(); cnt.zero_()
                    row.append(f"s{sp}M{mode}={t:.2f}")
        base = bench(lambda: x.float() @ state["router"].T) if is_r else bench(lambda: x @ state[layer][:n].T)
        lines.append(f"{layer[:4]} m{m} n{n} base {base:.2f} | " + " ".join(row) + f" [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
