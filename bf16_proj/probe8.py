# okc: problem=muse_bf16_proj
"""Probe 8, not a submission: no split-K at all for the BF16 skinny shapes. Narrow row tiles over
the full K per CTA (x as the 16/32-row MMA operand, W through TMA with deep pipelines), against
the current split-K swap-AB kernel. CUDA-graph replays from a cold L2."""

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
def _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, tile, SPLIT_K):
    """Write a unit's partial tile: directly, or into the workspace with the last unit reducing."""
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
        # All of this unit's adds land before it counts itself in; the last one in reduces.
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            # The others are done with this tile; read past L1 (the adds landed in L2), then re-zero.
            total = tl.load(ws_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=mask)
            # Every thread has read before any re-zeroes: the zeros can land in another thread's
            # layout than the loads (BM 128 on 8 warps lost partial sums without this).
            tl.debug_barrier()
            tl.store(ws_ptrs, tl.zeros_like(acc), mask=mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _swab_kernel(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[:, n-tile]^T = W[n-tile] @ x^T over this unit's k-slice; W and x through TMA."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    # This unit's share of the K chunks, balanced for any SPLIT_K.
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for c in range(c0, c1):
        x = x_desc.load([pid_m * BM, c * BK])  # rows past m read as zero
        w = w_desc.load([pid_n * BN, c * BK])
        acc = tl.dot(w, tl.trans(x), acc)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    ws_ptrs = ws_ptr + offs_m[None, :] * N + offs_n[:, None]
    _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K)



@triton.jit(do_not_specialize=["M"])
def _nosplit_kernel(w_desc, x_desc, out_ptr, M, N, K, stride_om,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, WS: tl.constexpr):
    pid_n = tl.program_id(0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for c in tl.range(0, K // BK, num_stages=STAGES, warp_specialize=WS):
        x = x_desc.load([0, c * BK])
        w = w_desc.load([pid_n * BN, c * BK])
        acc = tl.dot(x, tl.trans(w), acc)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=offs_m[:, None] < M)


def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    dev = state["router"].device
    ws = torch.zeros(WS_FLOATS, device=dev, dtype=torch.float32)
    cnt = torch.zeros(WS_TILES, device=dev, dtype=torch.int32)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]

    def bench(fn, reps=30):
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
    for m, layer in ((1, "unpooler"), (8, "unpooler"), (1, "qkv_gate"), (16, "qkv_gate"), (32, "qkv_gate"),
                     (1, "pooler"), (16, "pooler"), (32, "pooler"), (16, "unpooler"), (1, "o")):
        if time.time() - t0 > 265:
            lines.append("TIME BUDGET HIT")
            break
        w = state[layer]
        n, k = w.shape
        if layer == "unpooler" and m <= 8:
            n = 1024
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
        ref = x.double() @ w[:n].double().T
        key = "unpooler_slice" if layer == "unpooler" and n <= 1024 else layer
        bn0, bk0, wp0, st0, sp0 = BF16_CONFIGS[key]
        bm = max(16, triton.next_power_of_2(m))
        wd0 = TensorDescriptor.from_tensor(w[:n], [bn0, bk0])
        xd0 = TensorDescriptor.from_tensor(x, [bm, bk0])
        cur = bench(lambda: _swab_kernel[(n // bn0, sp0, 1)](wd0, xd0, out, ws, cnt, m, n, k, sp0, out.stride(0),
                                                              BM=bm, BN=bn0, BK=bk0, num_warps=wp0, num_stages=st0))
        res = []
        for bn in (16, 32):
            for bk in (128, 256):
                for st in (4, 6, 8, 10):
                    for wsp in (False, True):
                        if (bn + bm) * bk * 2 * st > 220 * 1024:
                            continue
                        wd = TensorDescriptor.from_tensor(w[:n], [bn, bk])
                        xd = TensorDescriptor.from_tensor(x, [bm, bk])
                        tag = f"{bn},{bk},{st},{'W' if wsp else '-'}"

                        def run(wd=wd, xd=xd, bn=bn, bk=bk, st=st, wsp=wsp):
                            _nosplit_kernel[(n // bn,)](wd, xd, out, m, n, k, out.stride(0), BM=bm, BN=bn, BK=bk,
                                                        STAGES=st, WS=wsp, num_warps=4)
                        try:
                            run()
                            err = float((out.double() - ref).norm() / ref.norm())
                            if err > 1e-2:
                                res.append((1e9, tag + f"BAD{err:.0e}"))
                                continue
                            res.append((bench(run), tag))
                        except Exception as e:  # noqa: BLE001
                            res.append((1e9, tag + type(e).__name__[:10]))
        res.sort()
        bad = [t for v, t in res if v >= 1e9]
        lines.append(f"{layer[:4]} m{m} n{n} cur {cur:.2f} (s{sp0}) | nosplit " + " ".join(f"{v:.2f}/{t}" for v, t in res[:5])
                     + f" | bad {len(bad)} {bad[0] if bad else ''} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
