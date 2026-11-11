// m = 65..256 projections on CUTLASS 3.x for Hopper: TMA + wgmma, warp-specialized cooperative
// kernel, with the stream-K tile scheduler so that split-K / stream-K spreads the few output
// tiles of these shapes over all SMs. Each config lives in its own translation unit (compiled in
// parallel); this header holds the shared template. SWAP puts W in the wgmma M slot
// (out^T = W @ x^T, written column-major = out row-major).
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template <bool SWAP, int TM, int TN, int TK, int CM>
struct ProjGemm {
  using T = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;     // K-major
  using LayoutB = cutlass::layout::ColumnMajor;  // K-major
  using LayoutD = cute::conditional_t<SWAP, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using ClusterShape = Shape<Int<CM>, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD, 8, T, LayoutD, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, T, LayoutA, 8, T, LayoutB, 8, float, TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue,
                                                      cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  // out [m, n] = x [m, k] @ w [n, k]^T. mode: 0 heuristic, 1 data-parallel, 2 split-K, 3 stream-K.
  static void run(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, int64_t splits,
                  int64_t mode, bool size_only, int64_t* ws_bytes) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int gm = SWAP ? n : m, gn = SWAP ? m : n;
    const T* a = reinterpret_cast<const T*>(SWAP ? w.data_ptr() : x.data_ptr());
    const T* b = reinterpret_cast<const T*>(SWAP ? x.data_ptr() : w.data_ptr());
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    auto sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(gm, k, 1));
    auto sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(gn, k, 1));
    auto sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(gm, gn, 1));
    auto sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(gm, gn, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    static int sms = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    hw.sm_count = sms;
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {gm, gn, k, 1},
                                  {a, sa, b, sb},
                                  {{1.f, 0.f}, nullptr, sc, reinterpret_cast<T*>(out.data_ptr()), sd},
                                  hw};
    using Decomp = cutlass::gemm::kernel::detail::DecompositionMode;
    args.scheduler.splits = (int)splits;
    args.scheduler.decomposition_mode = mode == 1 ? Decomp::DataParallel
                                        : mode == 2 ? Decomp::SplitK
                                        : mode == 3 ? Decomp::StreamK
                                                    : Decomp::Heuristic;
    Gemm gemm;
    if (size_only) {
      *ws_bytes = (int64_t)Gemm::get_workspace_size(args);
      return;
    }
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cutlass: cannot implement");
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "cutlass: init");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "cutlass: run");
  }
};

#define PROJ_GEMM_ENTRY(NAME, SWAP, TM, TN, TK, CM)                                                        \
  void NAME(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, int64_t splits,         \
            int64_t mode) {                                                                                \
    ProjGemm<SWAP, TM, TN, TK, CM>::run(w, x, out, ws, splits, mode, false, nullptr);                      \
  }                                                                                                        \
  int64_t NAME##_ws(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t splits, int64_t mode) {   \
    int64_t b = 0;                                                                                         \
    ProjGemm<SWAP, TM, TN, TK, CM>::run(w, x, out, out, splits, mode, true, &b);                           \
    return b;                                                                                              \
  }
