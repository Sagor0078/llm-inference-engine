# okc: problem=muse_moe_w4a16
"""Probe M5, not a submission: flashinfer.mm_bf16_fp4 (BF16 x NVFP4, group-16 E4M3 scales) on this sm90 GPU:
scale swizzle helper, per-backend prepare + correctness vs an exact dequantized reference + graph-replay timing
at M = 16 / 64 / 256 on one expert's w2 [4096, 4096]; and the docs of fused_moe's W4A16 / NVFP4 runners."""

import inspect
import time

import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant(packed, sf, gs):
    p = packed.view(torch.uint8)
    codes = torch.stack((p & 15, p >> 4), -1).flatten(-2)
    grid = torch.tensor(_E2M1, device=p.device)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    return (vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs).unsqueeze(-1)).view_as(vals)


def prepare(weights):
    t0 = time.time()
    import flashinfer
    lines = []
    fm = flashinfer.fused_moe
    for n in ("B12xW4A16Runner", "B12xW4A16Config", "CuTileNvfp4Runner", "B12xMoEWrapper"):
        obj = getattr(fm, n, None)
        lines.append(f"fused_moe.{n}: {(obj.__doc__ or '').strip()[:400].replace(chr(10), ' ') if obj else 'missing'}")
    helpers = [n for n in dir(flashinfer) if any(s in n.lower() for s in ("interleave", "swizzle", "scale"))]
    lines.append(f"scale helpers: {helpers[:30]}")
    dev = weights["w2"].device
    w = weights["w2"][0].contiguous()
    sf = weights["w2_sf"][0].contiguous()
    gs = weights["w2_gs"][0:1].contiguous()
    wd = _dequant(w, sf, float(gs))
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)

    def bench(fn, reps=20):
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        ts = []
        for _ in range(reps):
            torch.sum(flush, dim=0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gr.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    swz = None
    for n in ("block_scale_interleave", "nvfp4_block_scale_interleave", "fp4_swizzle_blockscale"):
        if hasattr(flashinfer, n):
            swz = getattr(flashinfer, n)
            lines.append(f"swizzle via {n}{inspect.signature(swz) if callable(swz) else ''}")
            break
    sf_sw = sf
    if swz is not None:
        try:
            sf_sw = swz(sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf)
            lines.append(f"swizzled sf {tuple(sf_sw.shape)} {sf_sw.dtype}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"swizzle FAILED {type(e).__name__}: {str(e)[:200]}")
    for backend in ("cudnn", "cute-dsl"):
        try:
            bp, sp, ap = flashinfer.prepare_bf16_fp4_weights(w, sf_sw, gs, backend=backend)
        except Exception as e:  # noqa: BLE001
            lines.append(f"{backend}: prepare FAILED {type(e).__name__}: {str(e)[:300]}")
            continue
        for m in (16, 64, 256):
            x = torch.randn((m, 4096), device=dev).to(torch.bfloat16)
            ref = x.float() @ wd.T
            try:
                fn = lambda x=x: flashinfer.mm_bf16_fp4(x, bp, sp, ap, backend=backend)
                out = fn().float()
                err = float((out - ref).norm() / ref.norm())
                t = bench(fn)
                dense = bench(lambda x=x: x @ wd.to(torch.bfloat16).T)
                lines.append(f"{backend} m{m}: err {err:.2e}  fp4 {t:.1f}us  (bf16 dense cuBLAS {dense:.1f}us)")
            except Exception as e:  # noqa: BLE001
                lines.append(f"{backend} m{m}: FAILED {type(e).__name__}: {str(e)[:300]}")
    lines.append(f"[{time.time() - t0:.0f}s]")
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
