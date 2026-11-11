# okc: problem=muse_bf16_proj
"""Probe 19, not a submission: which native Hopper GEMM toolchains the judge image offers
(TileLang, CUTLASS C++ headers bundled with flashinfer / deep_gemm / cutlass), and whether a
TileLang swap-AB split-K kernel (TMA + wgmma pipeline) beats cuBLAS at m=128/256 and our Triton at
m<=32, timed as cold-L2 CUDA-graph replays."""

import glob
import os
import time
import traceback

import torch


def _inventory(lines):
    import site
    roots = site.getsitepackages() + ["/usr/local/lib", "/usr/include", "/opt", "/usr/local/cuda/include"]
    hits = set()
    for r in roots:
        for pat in ("**/cutlass/gemm/collective/collective_builder.hpp", "**/cute/tensor.hpp",
                    "**/cutlass/gemm/kernel/tile_scheduler_params.h"):
            for p in glob.glob(os.path.join(r, pat), recursive=True):
                hits.add(p)
        if len(hits) > 12:
            break
    lines.append("headers: " + " | ".join(sorted(hits)[:12]))
    for p in sorted(hits):
        if p.endswith("collective_builder.hpp"):
            vh = p.split("/cutlass/gemm/")[0] + "/cutlass/version.h"
            if os.path.exists(vh):
                v = [l.strip() for l in open(vh) if "#define CUTLASS_" in l and ("MAJOR" in l or "MINOR" in l or "PATCH" in l)]
                lines.append(f"  {vh}: {v[:3]}")
    for modname in ("tilelang", "tvm", "cutlass", "sgl_kernel", "deep_gemm"):
        try:
            m = __import__(modname)
            lines.append(f"{modname} {getattr(m, '__version__', '?')} @ {os.path.dirname(m.__file__)}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"{modname} import failed: {type(e).__name__}: {str(e)[:160]}")
    try:
        import sgl_kernel
        lines.append("sgl_kernel gemm-ish: " + str([n for n in dir(sgl_kernel) if any(s in n.lower() for s in ("gemm", "mm", "bf16", "split"))][:40]))
    except Exception:  # noqa: BLE001
        pass


def _tl_kernel(N, K, MP, BN, BK, STAGES, SPLIT, THREADS):
    """out32[MP, N] += x[MP, kslice] @ W[n-tile, kslice]^T, as C^T tiles (W is the wgmma M operand)."""
    import tilelang
    import tilelang.language as T

    KS = K // SPLIT

    @T.prim_func
    def main(W: T.Tensor((N, K), "bfloat16"), X: T.Tensor((MP, K), "bfloat16"), O: T.Tensor((N, MP), "float32")):
        with T.Kernel(N // BN, SPLIT, threads=THREADS) as (bx, bz):
            w_s = T.alloc_shared((BN, BK), "bfloat16")
            x_s = T.alloc_shared((MP, BK), "bfloat16")
            acc = T.alloc_fragment((BN, MP), "float32")
            T.clear(acc)
            for kk in T.Pipelined(KS // BK, num_stages=STAGES):
                T.copy(W[bx * BN, bz * KS + kk * BK], w_s)
                T.copy(X[0, bz * KS + kk * BK], x_s)
                T.gemm(w_s, x_s, acc, transpose_B=True)
            if SPLIT == 1:
                T.copy(acc, O[bx * BN, 0])
            else:
                for i, j in T.Parallel(BN, MP):
                    T.atomic_add(O[bx * BN + i, j], acc[i, j])

    return tilelang.compile(main, target="cuda")


def prepare(weights):
    t0 = time.time()
    lines = [f"torch {torch.__version__} cuda {torch.version.cuda}"]
    _inventory(lines)
    print("\n".join(lines), flush=True)
    dev = weights["pooler"].device
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)

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
    # Pure streaming read of each weight (sum), for the roofline at this clock.
    for layer in ("pooler", "qkv_gate", "o"):
        w = weights[layer]
        s = torch.empty(w.shape[1], device=dev, dtype=torch.float32)
        lines.append(f"read {layer} sum0 {bench(lambda: torch.sum(w, dim=0, dtype=torch.float32, out=s)):.1f}")

    # (m, layer, [(BN, BK, STAGES, SPLIT, THREADS)])
    plan = [
        (128, "pooler", [(128, 64, 4, 4, 256), (64, 64, 4, 2, 128), (128, 64, 3, 2, 256), (64, 128, 3, 4, 128), (256, 64, 3, 8, 256)]),
        (128, "qkv_gate", [(128, 64, 4, 8, 256), (64, 64, 4, 4, 128), (64, 128, 3, 4, 128)]),
        (128, "unpooler", [(128, 64, 4, 2, 256), (64, 64, 4, 1, 128), (128, 64, 4, 4, 256)]),
        (128, "o", [(128, 64, 4, 2, 256), (64, 64, 4, 1, 128)]),
        (256, "pooler", [(128, 64, 3, 4, 256), (128, 64, 3, 2, 256), (64, 64, 3, 2, 128)]),
        (16, "pooler", [(64, 128, 4, 4, 128), (128, 128, 4, 8, 128), (128, 128, 4, 4, 128)]),
        (16, "qkv_gate", [(64, 128, 4, 8, 128), (128, 128, 4, 16, 128)]),
    ]
    for m, layer, cfgs in plan:
        w = weights[layer]
        n, k = w.shape
        mp = max(16, m)
        x = torch.randn((mp, k), device=dev).to(torch.bfloat16)
        ref = (x[:m].double() @ w.double().T)
        base = bench(lambda: x[:m] @ w.T)
        res = []
        for cfg in cfgs:
            if time.time() - t0 > 250:
                res.append("BUDGET")
                break
            bn, bk, st, sp, th = cfg
            try:
                tc = time.time()
                kern = _tl_kernel(n, k, mp, bn, bk, st, sp, th)
                ct = time.time() - tc
                o = torch.zeros((n, mp), device=dev, dtype=torch.float32)

                def run():
                    if sp > 1:
                        o.zero_()
                    kern(w, x, o)
                run()
                torch.cuda.synchronize()
                err = float((o.T[:m].double() - ref).norm() / ref.norm())
                t = bench(run)
                res.append(f"{t:.1f}/{bn},{bk},{st},s{sp},t{th} e{err:.0e} c{ct:.0f}s")
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()
                res.append(f"FAIL {cfg}: {type(e).__name__}: {str(e)[:300]} | {' / '.join(tb[-4:])[:500]}")
        lines.append(f"{layer} m{m} cublas {base:.1f} | " + " ; ".join(res))
        print(lines[-1], flush=True)
    lines.append(f"total {time.time() - t0:.0f}s")
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
