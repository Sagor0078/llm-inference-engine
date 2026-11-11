# okc: problem=muse_moe_w4a16
"""Probe D1, not a submission: CUTLASS C++ headers in the image (for load_inline), their version, and whether the sm90
mixed-input (narrow-B) GEMM collectives / grouped examples support E2M1 (FP4) weights with block scales."""

import glob
import os
import re
import time


def _grep(paths, pattern, limit=12):
    hits = []
    rx = re.compile(pattern)
    for p in paths:
        try:
            with open(p, errors="ignore") as f:
                for n, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append(f"{os.path.relpath(p, '/')}:{n}: {line.strip()[:150]}")
                        if len(hits) >= limit:
                            return hits
        except OSError:
            pass
    return hits


def prepare(weights):
    t0 = time.time()
    lines = []
    roots = set()
    for pat in ("/usr/local/lib/python3*/dist-packages/**/cutlass/version.h", "/opt/**/cutlass/version.h",
                "/usr/local/**/include/cutlass/version.h", "/usr/include/cutlass/version.h",
                "/root/**/cutlass/version.h"):
        for v in glob.glob(pat, recursive=True):
            roots.add(os.path.dirname(os.path.dirname(v)))
        if len(roots) > 6 or time.time() - t0 > 60:
            break
    lines.append(f"cutlass include roots ({len(roots)}): " + " | ".join(sorted(roots))[:1500])
    for root in sorted(roots)[:4]:
        ver = _grep([os.path.join(root, "cutlass/version.h")], r"#define CUTLASS_(MAJOR|MINOR|PATCH)", 3)
        mixed = sorted(glob.glob(os.path.join(root, "cutlass/gemm/collective/sm90_mma*mixed_input*.hpp")))
        lines.append(f"[{root}] version {ver}")
        lines.append(f"  sm90 mixed-input collectives: {[os.path.basename(m) for m in mixed]}")
        lines += ["  " + h for h in _grep(mixed, r"e2m1|float_e2m1|fp4|FP4", 8)]
        lines += ["  " + h for h in _grep(mixed, r"group_size|GroupSize|ScaleGranularity|chunk_size", 6)]
        util = glob.glob(os.path.join(root, "cutlass/detail/collective/mixed_input_utils.hpp"))
        lines += ["  util: " + h for h in _grep(util, r"e2m1|float_e2m1|int4b_t|LookupTable|lut", 10)]
        conv = glob.glob(os.path.join(root, "cutlass/numeric_conversion.h"))
        lines += ["  conv: " + h for h in _grep(conv, r"struct NumericArrayConverter<[^>]*bfloat16_t[^>]*float_e2m1_t", 4)]
        grouped = glob.glob(os.path.join(root, "../examples/69_hopper_mixed_dtype_grouped_gemm/*.cu"))
        lines.append(f"  example 69 files: {[os.path.basename(g) for g in grouped]}")
    lines.append(f"nvcc: {os.path.exists('/usr/local/cuda/bin/nvcc')}  [{time.time() - t0:.0f}s]")
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
