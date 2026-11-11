# okc: problem=muse_bf16_proj
"""Stress probe, not a submission: replays every split-K config thousands of times from a cold L2
under each reduction protocol and counts wrong outputs, then stops the run with the table."""

import time

import torch
import triton
import triton.language as tl

WS_FLOATS = 1 << 20
WS_TILES = 4096

# Per BF16 layer: [(max m, (BM, BN, BK, num_warps, num_stages, SPLIT_K, TILED))], BM 0 meaning the
# smallest MMA tile that covers m; past the last entry, cuBLAS. TILED reads the tile-major copy.
BF16_CONFIGS = {
    "qkv_gate": [(32, (0, 64, 128, 4, 3, 7, True))],
    "o": [(32, (0, 32, 128, 4, 4, 1, False))],
    "pooler": [(32, (0, 64, 128, 4, 3, 4, True)), (128, (128, 64, 64, 8, 4, 4, False))],
    "unpooler_slice": [(8, (0, 64, 128, 4, 4, 8, False))],  # m <= 8: this GPU's 1024-row slice
    "unpooler": [(16, (0, 64, 256, 4, 3, 1, True))],
}
# Tile-major copies, [N/BN, K/BK, BN, BK], for the configs that read them: (BN, BK) per layer.
TILED_LAYOUTS = {"qkv_gate": (64, 128), "pooler": (64, 128), "unpooler": (64, 256)}


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _proj_kernel(x_ptr, w_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K, SPLIT_K, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HI_LO: tl.constexpr, TILED: tl.constexpr, PROTO: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    # This unit's share of the K chunks, balanced for any SPLIT_K.
    chunks = K // BK
    c0 = pid_k * chunks // SPLIT_K
    c1 = (pid_k + 1) * chunks // SPLIT_K
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = c0 * BK + tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    if TILED:  # each unit's k-range of its tile is one contiguous run
        w_off = (pid_n * chunks + c0) * (BN * BK) + tl.arange(0, BN)[:, None] * BK + tl.arange(0, BK)[None, :]
    else:
        w_off = offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(c0, c1):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptr + w_off, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
        if HI_LO:
            lo = tl.load(lo_ptr + w_off, eviction_policy="evict_first")
            acc = tl.dot(x, tl.trans(lo), acc)
        x_ptrs += BK
        w_off += BN * BK if TILED else BK

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask)
    else:
        ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.atomic_add(ws_ptrs, acc, mask=m_mask, sem="relaxed")
        # All of this unit's adds land before it counts itself in; the last one in reduces.
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(0) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            if PROTO == 2:
                total = tl.atomic_xchg(ws_ptrs, tl.zeros((BM, BN), dtype=tl.float32), mask=m_mask, sem="acquire")
                tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask)
            else:
                total = tl.load(ws_ptrs, mask=m_mask, other=0.0, cache_modifier=".cg")
                tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask)
                if PROTO == 1:
                    tl.debug_barrier()
                tl.store(ws_ptrs, tl.zeros((BM, BN), dtype=tl.float32), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed")


def _config(layer: str, m: int, n: int, k: int):
    """(BM, BN, BK, num_warps, num_stages, SPLIT_K, TILED) for a shape, or None for cuBLAS. Shape-only."""
    if n % 64 or k % 256:
        return None
    small_bm = max(16, triton.next_power_of_2(m))
    if layer == "router":
        if m <= 16:
            return (small_bm, 32, 128, 4, 4, 16, False)
        if m <= 32:
            return (small_bm, 64, 128, 4, 4, 33, False)
        if m <= 128:
            return (64, 32, 128, 4, 3, 16, False)
        tiles = triton.cdiv(m, 64) * (n // 64)
        return (64, 64, 128, 4, 4, max(1, min(8, 264 // tiles)), False)
    key = "unpooler_slice" if layer == "unpooler" and n <= 1024 else layer
    for max_m, cfg in BF16_CONFIGS.get(key, ()):
        if m <= max_m:
            return (cfg[0] or small_bm,) + cfg[1:]
    return None


def _tile(w: torch.Tensor, bn: int, bk: int) -> torch.Tensor:
    """[N, K] -> [N/BN, K/BK, BN, BK]: the same values, tile-major (lossless repack)."""
    n, k = w.shape
    return w.reshape(n // bn, bn, k // bk, bk).permute(0, 2, 1, 3).contiguous()



def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    for layer, (bn, bk) in TILED_LAYOUTS.items():
        state[f"_tiled_{layer}"] = _tile(state[layer], bn, bk)
    hi = state["router"].to(torch.bfloat16)
    lo = (state["router"] - hi.float()).to(torch.bfloat16)
    dev = hi.device
    ws = torch.zeros(WS_FLOATS, device=dev, dtype=torch.float32)
    cnt = torch.zeros(WS_TILES, device=dev, dtype=torch.int32)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.float32)
    sink = torch.empty((), device=dev)
    lines = [f"torch {torch.__version__} triton {triton.__version__}"]
    shapes = [(128, "pooler"), (100, "pooler"), (1, "qkv_gate"), (32, "qkv_gate"), (8, "pooler"), (32, "pooler"),
              (1, "unpooler"), (8, "unpooler"), (1, "router"), (16, "router"), (32, "router"), (128, "router"),
              (256, "router"), (1000, "router")]
    reps = 1500
    for proto in (0, 1, 2):
        for m, layer in shapes:
            if time.time() - t0 > 270:
                lines.append("TIME BUDGET HIT")
                break
            n, k = weights[layer].shape
            if layer == "unpooler" and m <= 8:
                n = 1024
            cfg = _config(layer, m, n, k)
            bm, bn, bk, warps, stages, split, tiled = cfg
            is_r = layer == "router"
            w = hi if is_r else (state[f"_tiled_{layer}"] if tiled else state[layer])
            lo_ = lo if is_r else w
            x = torch.randn((m, k), device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), device=dev, dtype=torch.float32 if is_r else torch.bfloat16)
            ref = (x.double() @ state[layer][:n].double().T).float()
            rn = ref.norm(dim=1).clamp(min=1e-6)
            grid = (n // bn, split, triton.cdiv(m, bm))

            def run():
                _proj_kernel[grid](x, w, lo_, out, ws, cnt, m, n, k, split, x.stride(0), k, out.stride(0),
                                   BM=bm, BN=bn, BK=bk, HI_LO=is_r, TILED=tiled, PROTO=proto,
                                   num_warps=warps, num_stages=stages)
            try:
                run()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run()
                bad = torch.zeros((), device=dev, dtype=torch.int32)
                worst = torch.zeros((), device=dev)
                for i in range(reps):
                    if i % 2 == 0:
                        torch.sum(flush, dim=0, out=sink)
                    g.replay()
                    err = ((out.float() - ref).norm(dim=1) / rn).max()
                    bad += (err > 5e-3).int()
                    torch.maximum(worst, err, out=worst)
                torch.cuda.synchronize()
                lines.append(f"P{proto} {layer[:4]} m{m} cfg{cfg} bad {int(bad)}/{reps} worst {float(worst):.1e} "
                             f"wsclean {int(ws.abs().max() == 0)} cntclean {int(cnt.abs().max() == 0)} [{time.time() - t0:.0f}s]")
            except Exception as e:  # noqa: BLE001
                lines.append(f"P{proto} {layer[:4]} m{m} EXC {type(e).__name__}: {str(e)[:120]}")
            ws.zero_(); cnt.zero_()
    raise RuntimeError("STRESS\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("stress probe")
