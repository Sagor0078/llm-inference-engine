# okc: problem=muse_bf16_proj
"""Probe 9, not a submission: is native CUDA via torch's load_inline usable in the judge (the
documented path), how long does it compile, and what DeepGEMM / CUTLASS offer for BF16."""

import os
import time

import torch

CUDA_SRC = r"""
#include <cuda_bf16.h>
__global__ void cluster_probe(const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(x[i]) * 2.0f;
}
void launch(torch::Tensor x, torch::Tensor out) {
  int n = x.numel();
  cluster_probe<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), n);
}
"""
CPP_SRC = "void launch(torch::Tensor x, torch::Tensor out);"


def prepare(weights):
    lines = [f"torch {torch.__version__} cuda {torch.version.cuda}"]
    t0 = time.time()
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load_inline
        lines.append(f"CUDA_HOME {CUDA_HOME}")
        mod = load_inline(name="okc_probe9", cpp_sources=CPP_SRC,
                          cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC,
                          functions=["launch"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          verbose=False)
        x = torch.randn(1000, device="cuda").to(torch.bfloat16)
        out = torch.empty(1000, device="cuda")
        mod.launch(x, out)
        torch.cuda.synchronize()
        ok = torch.allclose(out, x.float() * 2)
        lines.append(f"load_inline OK={ok} compile {time.time() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        lines.append(f"load_inline FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {str(e)[:600]}")
    for modname in ("deep_gemm", "cutlass", "cutlass.cute", "flashinfer", "tilelang", "sgl_kernel"):
        try:
            m = __import__(modname, fromlist=["*"])
            names = [n for n in dir(m) if any(s in n.lower() for s in ("bf16", "gemm", "gemv", "mm"))]
            lines.append(f"{modname} {getattr(m, '__version__', '?')}: {names[:25]}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"{modname} import failed: {type(e).__name__}: {str(e)[:120]}")
    lines.append(f"nproc {os.cpu_count()}")
    raise RuntimeError("PROBE\n" + "\n".join(lines))


def kernel_fn(data, state):
    raise RuntimeError("probe")
