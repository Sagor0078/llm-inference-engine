# okc: problem=muse_bf16_proj
"""Tuning probe 5, not a submission: m=128/256 BF16 shapes. Normal (x as the 128-row MMA operand,
wide N) and swap-AB TMA kernels, with and without warp specialization, against cuBLAS and SGLang's
own dispatch, as CUDA-graph replays from a cold L2."""

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


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _router_kernel(x_ptr, hi_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_xm, stride_wn,
                   stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out = x @ (hi + lo)^T in FP32, both BF16 halves through the tensor cores."""
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
    _finish(acc, out_ptrs, ws_ptrs, m_mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K)



@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _normal_kernel(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_om,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, WS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for c in tl.range(c0, c1, num_stages=STAGES, warp_specialize=WS):
        x = x_desc.load([pid_m * BM, c * BK])
        w = w_desc.load([pid_n * BN, c * BK])
        acc = tl.dot(x, tl.trans(w), acc)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = offs_m[:, None] < M
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
    _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K)


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _swab_ws_kernel(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_om,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, WS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for c in tl.range(c0, c1, num_stages=STAGES, warp_specialize=WS):
        x = x_desc.load([pid_m * BM, c * BK])
        w = w_desc.load([pid_n * BN, c * BK])
        acc = tl.dot(w, tl.trans(x), acc)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    ws_ptrs = ws_ptr + offs_m[None, :] * N + offs_n[:, None]
    _finish(acc, out_ptrs, ws_ptrs, mask, cnt_ptr, pid_m * tl.num_programs(0) + pid_n, SPLIT_K)


SMS = 132


def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    dev = state["router"].device
    ws = torch.zeros(1 << 22, device=dev, dtype=torch.float32)
    cnt = torch.zeros(WS_TILES, device=dev, dtype=torch.int32)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]
    try:
        from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl as sgl
    except Exception as e:  # noqa: BLE001
        sgl = None
        lines.append(f"no sglang dispatch: {type(e).__name__}")

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
    shapes = [(128, l) for l in ("pooler", "unpooler", "qkv_gate", "o")] + [(256, l) for l in ("pooler", "unpooler", "qkv_gate", "o")]
    for m, layer in shapes:
        if time.time() - t0 > 265:
            lines.append("TIME BUDGET HIT")
            break
        w = state[layer]
        n, k = w.shape
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
        ref = x.double() @ w.double().T
        base = bench(lambda: x @ w.T)
        sg = bench(lambda: sgl(x, w, None)) if sgl is not None else float("nan")
        res = []
        # (kind, BM, BN, BK, warps, stages, ws)
        space = []
        for bn in (128, 256):
            for bk in (64, 128):
                for st in (3, 4):
                    for wsp in (False, True):
                        smem = (m * bk + bn * bk) * 2 * st
                        if smem <= 220 * 1024 and m * bn // 256 <= 160:
                            space.append(("N", m if m <= 256 else 256, bn, bk, 8, st, wsp))
        for bn in (64, 128):
            for bk in (64, 128):
                for st in (3, 4):
                    for wsp in (False, True):
                        wp = 4 if bn == 64 else 8
                        smem = (m * bk + bn * bk) * 2 * st
                        if smem <= 220 * 1024 and m * bn // (wp * 32) <= 128:
                            space.append(("S", m, bn, bk, wp, st, wsp))
        for kind, bm, bn, bk, warps, stages, wsp in space:
            if time.time() - t0 > 265:
                break
            tiles = triton.cdiv(m, bm) * (n // bn)
            chunks = k // bk
            wd = TensorDescriptor.from_tensor(w, [bn, bk])
            xd = TensorDescriptor.from_tensor(x, [bm, bk])
            kern = _normal_kernel if kind == "N" else _swab_ws_kernel
            for split in sorted({max(1, min(chunks, round(t / tiles))) for t in (tiles, SMS, 2 * SMS)}):
                if split > 1 and m * n > ws.numel():
                    continue
                grid = (n // bn, split, triton.cdiv(m, bm))

                def run(kern=kern, grid=grid, wd=wd, xd=xd, split=split, bm=bm, bn=bn, bk=bk, warps=warps, stages=stages, wsp=wsp):
                    kern[grid](wd, xd, out, ws, cnt, m, n, k, split, out.stride(0), BM=bm, BN=bn, BK=bk,
                               STAGES=stages, WS=wsp, num_warps=warps)
                tag = f"{kind}{bm},{bn},{bk},{warps},{stages},{'W' if wsp else '-'},s{split}"
                try:
                    run()
                    err = float((out.double() - ref).norm() / ref.norm())
                    if err > 1e-2 or cnt.abs().max() != 0:
                        res.append((1e9, tag + f"BAD{err:.0e}"))
                        continue
                    res.append((bench(run), tag))
                except Exception as e:  # noqa: BLE001
                    res.append((1e9, tag + type(e).__name__[:10]))
        res.sort()
        bad = [t for v, t in res if v >= 1e9]
        lines.append(f"{layer[:4]} m{m} torch {base:.1f} sgl {sg:.1f} | " + " ".join(f"{v:.1f}/{t}" for v, t in res[:5])
                     + f" | bad {len(bad)} {bad[0] if bad else ''} [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    raise RuntimeError("TUNE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("tuning probe")
