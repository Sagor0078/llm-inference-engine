# okc: problem=muse_bf16_proj
"""Probe 10, not a submission: DeepGEMM / FlashInfer BF16 GEMMs vs cuBLAS on the m=128/256 shapes,
as CUDA-graph replays from a cold L2. Signatures inspected and printed; only correct results timed."""

import inspect
import time

import torch


def prepare(weights):
    t0 = time.time()
    dev = weights["router"].device
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = []

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

    import deep_gemm
    import flashinfer
    for name, fn in (("deep_gemm.bf16_gemm_nt", deep_gemm.bf16_gemm_nt), ("deep_gemm.cublaslt_gemm_nt", deep_gemm.cublaslt_gemm_nt),
                     ("flashinfer.mm_bf16", flashinfer.mm_bf16)):
        try:
            sig = str(inspect.signature(fn))
        except Exception:  # noqa: BLE001
            sig = "?"
        doc = (fn.__doc__ or "").strip().replace("\n", " ")[:300]
        lines.append(f"{name}{sig} | {doc}")
    nop = torch.zeros(1, device=dev)
    lines.append(f"nop {bench(lambda: nop.add_(1)):.2f}")
    for m in (128, 256):
        for layer in ("qkv_gate", "o", "pooler", "unpooler"):
            if time.time() - t0 > 250:
                break
            w = weights[layer].contiguous()
            n, k = w.shape
            x = torch.randn((m, k), device=dev).to(torch.bfloat16)
            ref = x.double() @ w.double().T
            out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
            row = [f"torch={bench(lambda: torch.mm(x, w.T, out=out)):.2f}"]
            cands = {
                "dg_nt": lambda: deep_gemm.bf16_gemm_nt(x, w, out),
                "dg_lt": lambda: deep_gemm.cublaslt_gemm_nt(x, w, out),
                "fi_mm": lambda: flashinfer.mm_bf16(x, w.T, out=out),
                "fi_mm_cutlass": lambda: flashinfer.mm_bf16(x, w.T, out=out, backend="cutlass"),
                "fi_mm_cudnn": lambda: flashinfer.mm_bf16(x, w.T, out=out, backend="cudnn"),
            }
            for cname, fn in cands.items():
                try:
                    out.zero_()
                    fn()
                    torch.cuda.synchronize()
                    err = float((out.double() - ref).norm() / ref.norm())
                    if err > 1e-2:
                        row.append(f"{cname}=BAD{err:.0e}")
                        continue
                    row.append(f"{cname}={bench(fn):.2f}")
                except Exception as e:  # noqa: BLE001
                    row.append(f"{cname}=EXC:{type(e).__name__}:{str(e)[:90]}")
            lines.append(f"{layer[:4]} m{m} | " + " ".join(row) + f" [{time.time() - t0:.0f}s]")
            print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
