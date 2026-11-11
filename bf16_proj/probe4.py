# okc: problem=muse_bf16_proj
"""Roofline probe, not a submission: the fastest pure streaming read of each weight's bytes, and the
cost of decoding a 12-bit (8-bit low byte + 4-bit high-byte code) weight format in-kernel."""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _read_kernel(src, out, per_prog, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * per_prog
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(0, per_prog, BLOCK):
        acc ^= tl.load(src + base + i + tl.arange(0, BLOCK), eviction_policy="evict_first")
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


@triton.jit
def _decode_kernel(low_ptr, code_ptr, lut_ptr, out, per_prog, BLOCK: tl.constexpr, MODE: tl.constexpr):
    """per_prog weights per program: low bytes (uint8) and 4-bit codes (2 per uint8)."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * per_prog
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, per_prog, BLOCK):
        lo = tl.load(low_ptr + base + i + tl.arange(0, BLOCK), eviction_policy="evict_first").to(tl.int32)
        cb = tl.load(code_ptr + (base + i) // 2 + tl.arange(0, BLOCK // 2), eviction_policy="evict_first").to(tl.int32)
        c = tl.interleave(cb & 15, cb >> 4)
        if MODE == 0:  # arithmetic: sign bit + 3-bit level above a base
            hb = ((c & 8) << 4) | ((c & 7) + 56)
        else:  # 16-entry LUT gather
            hb = tl.load(lut_ptr + c)
        bits = ((hb << 8) | lo).to(tl.int16)
        w = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc += w
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


def prepare(weights):
    t0 = time.time()
    dev = weights["router"].device
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]

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
    out = torch.empty(1 << 24, device=dev, dtype=torch.int32)
    outf = torch.empty(1 << 24, device=dev, dtype=torch.float32)
    # Exponent statistics of the real weights: distinct high bytes and coverage of the top 15.
    for name, w in weights.items():
        if w.dtype == torch.bfloat16:
            hb = (w.view(torch.int16).to(torch.int32) >> 8) & 255
            counts = torch.bincount(hb.flatten(), minlength=256)
            top = counts.sort(descending=True).values
            lines.append(f"{name} distinct_hb {int((counts > 0).sum())} top15_cover {float(top[:15].sum() / counts.sum()):.6f} "
                         f"top14 {float(top[:14].sum() / counts.sum()):.6f} zeros {int((w == 0).sum())}")
    for label, nbytes in (("pool/unpo", 4096 * 8192 * 2), ("qkv", 2304 * 8192 * 2), ("o/router", 8192 * 1024 * 2),
                          ("slice", 1024 * 4096 * 2), ("pool12bit", 4096 * 8192 * 3 // 2)):
        src = torch.randint(-2**31, 2**31 - 1, (nbytes // 4,), device=dev, dtype=torch.int32)
        n_int = nbytes // 4
        res = []
        for block in (512, 1024, 2048, 4096):
            for warps in (4, 8):
                for stages in (1, 3):
                    for progs in (128, 256, 512, 1024, 2048):
                        per = n_int // progs
                        if per % block or per < block:
                            continue
                        try:
                            t = bench(lambda: _read_kernel[(progs,)](src, out, per, BLOCK=block, num_warps=warps, num_stages=stages))
                            res.append((t, block, warps, stages, progs))
                        except Exception as e:  # noqa: BLE001
                            res.append((1e9, block, warps, stages, type(e).__name__[:8]))
        res.sort()
        best = res[0][0]
        lines.append(f"read {label} {nbytes / 1e6:.1f}MB best {best:.1f}us ({nbytes / ((best - 5.3) * 1e-6) / 1e12:.2f} TB/s net of nop) "
                     + " ".join(f"{t:.1f}/{b},{w},{s},{p}" for t, b, w, s, p in res[:4]) + f" [{time.time() - t0:.0f}s]")
        print(lines[-1], flush=True)
    # 12-bit decode cost at pooler size (33.5M weights): compare with the pure read of the same 50 MB.
    nw = 4096 * 8192
    low = torch.randint(0, 255, (nw,), device=dev, dtype=torch.uint8)
    codes = torch.randint(0, 255, (nw // 2,), device=dev, dtype=torch.uint8)
    lut = torch.arange(16, device=dev, dtype=torch.int32) + 48
    for mode in (0, 1):
        res = []
        for block in (1024, 2048, 4096):
            for warps in (4, 8):
                for progs in (256, 512, 1024, 2048):
                    per = nw // progs
                    if per % block:
                        continue
                    try:
                        t = bench(lambda: _decode_kernel[(progs,)](low, codes, lut, outf, per, BLOCK=block, MODE=mode, num_warps=warps))
                        res.append((t, block, warps, progs))
                    except Exception as e:  # noqa: BLE001
                        res.append((1e9, block, warps, type(e).__name__[:8] + str(e)[:80]))
        res.sort()
        lines.append(f"decode12 mode{mode} best {res[0][0]:.1f}us " + " ".join(f"{t:.1f}/{b},{w},{p}" for t, b, w, p in res[:4]))
        print(lines[-1], flush=True)
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
