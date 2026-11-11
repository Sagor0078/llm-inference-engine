# okc: problem=muse_bf16_proj
"""Skinny projections as split-K Triton kernels.

Every scored shape is bandwidth-bound (m <= 256), so the kernels are built to stream W once at
full HBM rate: (n-tile, k-slice) work units sized to fill the SMs in one wave, partial sums
reduced in-kernel by fp32 atomics into a persistent zeroed workspace, and the last unit to
finish a tile writes the output and re-zeroes the workspace, so a call is a single launch.

BF16 layers (m <= 32) load through TMA with the weights as the wide MMA operand ("swap AB":
out^T = W @ x^T), so a 64-row weight tile feeds the tensor cores against all of m at once.

The router's FP32 weights run as a BF16 hi + lo pair (split once in prepare, as SGLang does
above 8 tokens) with both halves through the tensor cores into one FP32 accumulator, in one
pass over the weights instead of two GEMMs and an add, at every m.

Above 32 tokens the BF16 layers stay on cuBLAS, which nothing here beat there, except the
pooler at 65-128 tokens (tune3.py, tune5.py).
Configs come from sweeping CUDA-graph replays from a cold L2 on the judge's H200.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


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


def _router_config(m: int, n: int):
    """(BM, BN, BK, num_warps, num_stages, SPLIT_K) for the router at m tokens. Shape-only."""
    if m <= 16:
        return (16, 32, 128, 4, 4, 16)
    if m <= 32:
        return (32, 64, 128, 4, 4, 33)
    if m <= 128:
        return (64, 32, 128, 4, 3, 16)
    tiles = triton.cdiv(m, 64) * (n // 64)
    return (64, 64, 128, 4, 4, max(1, min(8, 264 // tiles)))


def _workspace(state, m: int, n: int, tiles: int, split: int, device):
    if split == 1 or (m * n <= WS_FLOATS and tiles <= WS_TILES):
        return state["_ws"], state["_cnt"]
    # Outside the persistent workspace's reach: a fresh zeroed one (shape-based, graph-safe).
    return (torch.zeros(m * n, device=device, dtype=torch.float32),
            torch.zeros(tiles, device=device, dtype=torch.int32))


def _router(x: torch.Tensor, n: int, state) -> torch.Tensor:
    m, k = x.shape
    bm, bn, bk, warps, stages, split = _router_config(m, n)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    tiles = triton.cdiv(m, bm) * (n // bn)
    ws, cnt = _workspace(state, m, n, tiles, split, x.device)
    hi, lo = state["_router_hi"], state["_router_lo"]
    _router_kernel[(n // bn, split, triton.cdiv(m, bm))](
        x, hi, lo, out, ws, cnt, m, n, k, split, x.stride(0), hi.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def _swab(x: torch.Tensor, layer: str, n: int, state) -> torch.Tensor:
    m, k = x.shape
    key = "unpooler_slice" if layer == "unpooler" and n <= 1024 else layer
    if m > SKINNY_MAX_M:
        key = "pooler_m128"
    bn, bk, warps, stages, split = BF16_CONFIGS[key]
    bm = max(16, triton.next_power_of_2(m))
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    tiles = n // bn
    ws, cnt = _workspace(state, m, n, tiles, split, x.device)
    w_desc = state["_desc"][(layer, n)]
    x_desc = TensorDescriptor.from_tensor(x, [bm, bk])
    _swab_kernel[(n // bn, split, 1)](w_desc, x_desc, out, ws, cnt, m, n, k, split, out.stride(0),
                                       BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def prepare(weights: Weights):
    """Weights as given, their TMA descriptors, the router's BF16 hi + lo split, a zeroed split-K
    workspace, and every kernel compiled ahead of the timed calls."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["_router_hi"] = hi
    state["_router_lo"] = (router - hi.float()).to(torch.bfloat16)
    device = router.device
    state["_ws"] = torch.zeros(WS_FLOATS, device=device, dtype=torch.float32)
    state["_cnt"] = torch.zeros(WS_TILES, device=device, dtype=torch.int32)
    state["_desc"] = {}
    for key, (bn, bk, *_) in BF16_CONFIGS.items():
        layer = key.split("_")[0] if key != "qkv_gate" else key
        n = 1024 if key == "unpooler_slice" else state[layer].shape[0]
        state["_desc"][(layer, n)] = TensorDescriptor.from_tensor(state[layer][:n], [bn, bk])
    for layer, w in weights.items():
        n_full, k = w.shape
        for m in (1, 5, 8, 9, 16, 32, 33, 64, 128, 256, 512, 1000):
            n = 1024 if layer == "unpooler" and m <= 8 else n_full
            x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
            kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    """x @ W[:n].T: bfloat16, or float32 for the router."""
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    if x.stride(1) != 1 or x.data_ptr() % 16:
        x = x.contiguous().clone()
    m, k = x.shape
    if layer == "router":
        if n % 64 or k % 256:
            return x.float() @ state["router"][:n].T
        return _router(x, n, state)
    swab = m <= SKINNY_MAX_M or (layer == "pooler" and 64 < m <= 128)
    if swab and (layer, n) in state["_desc"] and k % 256 == 0:
        return _swab(x, layer, n, state)
    return x @ state[layer][:n].T

@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _router_tma(hd, ld, xd, out, ws, cnt, M, N: tl.constexpr, K: tl.constexpr, SPLIT_K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn, pk, pm = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    chunks = K // BK
    c0, c1 = pk * chunks // SPLIT_K, (pk + 1) * chunks // SPLIT_K
    acc = tl.zeros((BN, BM), tl.float32)
    for c in range(c0, c1):
        x = xd.load([pm * BM, c * BK])
        h = hd.load([pn * BN, c * BK])
        l = ld.load([pn * BN, c * BK])
        acc = tl.dot(h, tl.trans(x), acc)
        acc = tl.dot(l, tl.trans(x), acc)
    rn = pn * BN + tl.arange(0, BN)
    rm = pm * BM + tl.arange(0, BM)
    off = rm[None, :] * N + rn[:, None]
    _finish(acc, out + off, ws + off, rm[None, :] < M, cnt, pm * tl.num_programs(0) + pn, SPLIT_K)

_original_prepare = prepare
def prepare(weights):
    import time
    start = time.time()
    state = _original_prepare(weights)
    dev = weights['router'].device
    flush = torch.zeros(64 << 20, device=dev)
    sink = torch.empty((), device=dev)
    def bench(fn):
        fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        times=[]
        for _ in range(25):
            torch.sum(flush, out=sink, dim=0)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); b.synchronize()
            times.append(a.elapsed_time(b)*1000)
        return sorted(times)[12]
    for m in (128,256):
        for layer in ('qkv_gate','o','pooler','unpooler'):
            w=state[layer]; n,k=w.shape
            wt=w.T.contiguous()
            x=torch.randn((m,k),device=dev,dtype=torch.bfloat16)
            out=torch.empty((m,n),device=dev,dtype=torch.bfloat16)
            old=lambda: kernel_fn({'x':x,'config':{'layer':layer,'n':n}},state)
            nn=lambda: torch.mm(x,wt,out=out)
            err=((nn().float()-(x@w.T).float()).norm()/(x@w.T).float().norm()).item()
            print('TRANSPOSE',m,layer,bench(old),bench(nn),err,flush=True)
    for m in (1,8,16,32,128,256):
        n,k=weights['router'].shape
        x=torch.randn((m,k),device=dev,dtype=torch.bfloat16)
        out=torch.empty((m,n),device=dev,dtype=torch.float32)
        ref=x.double()@weights['router'].double().T
        old=bench(lambda:_router(x,n,state))
        results=[]
        for bm in ([16] if m<=16 else [32] if m==32 else [32,64]):
          for bn,bk,st in ((32,128,3),(64,128,3),(64,256,3),(32,256,3),(64,128,4)):
            hd=TensorDescriptor.from_tensor(state['_router_hi'],[bn,bk])
            ld=TensorDescriptor.from_tensor(state['_router_lo'],[bn,bk])
            xd=TensorDescriptor.from_tensor(x,[bm,bk])
            tiles=triton.cdiv(m,bm)*(n//bn)
            for target in (132,264):
              split=min(k//bk,max(1,target//tiles))
              try:
                fn=lambda:_router_tma[(n//bn,split,triton.cdiv(m,bm))](hd,ld,xd,out,state['_ws'],state['_cnt'],m,n,k,split,BM=bm,BN=bn,BK=bk,num_warps=4,num_stages=st)
                fn()
                err=((out.double()-ref).norm()/ref.norm()).item()
                if err>1e-4: print('BAD',m,bm,bn,bk,split,err,flush=True); continue
                t=bench(fn)
                results.append((round(t,3),bm,bn,bk,st,split))
              except Exception as e: print('ERR',str(e)[:150],flush=True)
        print('ROUTER',m,'old',old,'best',sorted(results)[:8],'elapsed',time.time()-start,flush=True)
    return state
