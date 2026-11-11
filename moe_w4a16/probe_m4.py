# okc: problem=muse_moe_w4a16
"""Probe M4, not a submission: which W4A16 / FP4 kernels in the image accept BF16 activations x NVFP4 weights on
this H200 (sm90), and Marlin's full signature. Prints signatures and docs, then tries each candidate on one
expert's weights (E2M1 + E4M3 group-16 scales + FP32 global scale) against a dequantized FP32 reference."""

import inspect
import time

import torch


def _sig(fn):
    try:
        return str(inspect.signature(fn))
    except Exception:  # noqa: BLE001
        return "?"


def _doc(fn, n=500):
    return (getattr(fn, "__doc__", None) or "").strip().replace("\n", " ")[:n]


def prepare(weights):
    t0 = time.time()
    lines = [f"torch {torch.__version__} cap {torch.cuda.get_device_capability()}"]
    names = []
    try:
        import flashinfer
        for n in ("mm_bf16_fp4", "prepare_bf16_fp4_weights", "CuteDslBf16MoEWrapper", "cute_dsl_fused_moe_bf16",
                  "grouped_mm_fp4", "mm_fp4", "fused_moe"):
            names.append((f"flashinfer.{n}", getattr(flashinfer, n, None)))
        fm = getattr(flashinfer, "fused_moe", None)
        if fm is not None:
            lines.append("flashinfer.fused_moe members: " + ", ".join(m for m in dir(fm) if not m.startswith("_"))[:600])
    except Exception as e:  # noqa: BLE001
        lines.append(f"flashinfer import: {e}")
    try:
        import deep_gemm
        for n in ("bf16_mega_moe", "fp8_fp4_gemm_nt", "bf16_m_grouped_gemm_nt_masked"):
            names.append((f"deep_gemm.{n}", getattr(deep_gemm, n, None)))
    except Exception as e:  # noqa: BLE001
        lines.append(f"deep_gemm import: {e}")
    try:
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        names.append(("moe_wna16_marlin_gemm", moe_wna16_marlin_gemm))
    except Exception as e:  # noqa: BLE001
        lines.append(f"marlin import: {e}")
    for n, fn in names:
        if fn is None:
            lines.append(f"{n}: missing")
        else:
            lines.append(f"{n}{_sig(fn)} :: {_doc(fn)}")
    # Try flashinfer's BF16 x FP4 dense GEMM on one expert's w2 [4096, 4096/2].
    dev = weights["w2"].device
    try:
        import flashinfer
        sys_x = torch.randn((64, 4096), device=dev).to(torch.bfloat16)
        w = weights["w2"][0].contiguous()
        sf = weights["w2_sf"][0].contiguous()
        gs = weights["w2_gs"][0]
        prep = getattr(flashinfer, "prepare_bf16_fp4_weights", None)
        if prep is not None:
            try:
                pw = prep(w, sf) if len(inspect.signature(prep).parameters) >= 2 else prep(w)
                lines.append(f"prepare_bf16_fp4_weights ok: {type(pw)} {[getattr(t, 'shape', None) for t in (pw if isinstance(pw, (tuple, list)) else [pw])]}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"prepare_bf16_fp4_weights FAILED: {type(e).__name__}: {str(e)[:300]}")
        mm = getattr(flashinfer, "mm_bf16_fp4", None)
        if mm is not None:
            for call in ("mm(x, w, sf, gs)", "mm(x, w.T, sf, gs)"):
                try:
                    out = eval(call, {"mm": mm, "x": sys_x, "w": w, "sf": sf, "gs": gs})
                    lines.append(f"{call}: ok {tuple(out.shape)} {out.dtype}")
                except Exception as e:  # noqa: BLE001
                    lines.append(f"{call}: FAILED {type(e).__name__}: {str(e)[:300]}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"flashinfer trial: {type(e).__name__}: {str(e)[:300]}")
    lines.append(f"[{time.time() - t0:.0f}s]")
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
