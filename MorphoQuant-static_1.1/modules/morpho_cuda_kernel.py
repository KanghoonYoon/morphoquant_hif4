"""
Fused CUDA kernel for MorphoQuant activation quantization + dequantization.

Replaces the multi-launch PyTorch element-wise path::

    x.float() → /scale → +zero → round → clamp → -zero → *scale → .to(input_dtype)

with a single CUDA kernel launch.  Also handles sparse outlier bypass
(bias compensation) for channels where ``|x[ch]| > compensation_limit[ch]``
and ``outlier_mask[ch] == True``.

Uses ``torch.utils.cpp_extension.load_inline`` for JIT compilation at
import time.  Falls back to the PyTorch path gracefully if CUDA toolkit
is unavailable.
"""

import torch
import warnings
from typing import Optional

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_morpho_cuda_module = None
_load_attempted = False


# ---------------------------------------------------------------------------
# CUDA kernel source (compiled with nvcc)
# ---------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

// -------------------------------------------------------------------------
// Type traits for explicit float ↔ half/bf16 conversion.
//
// PyTorch compiles with __CUDA_NO_HALF_CONVERSIONS__ and
// __CUDA_NO_BFLOAT16_CONVERSIONS__, which remove implicit conversion
// operators.  We use the hardware intrinsics explicitly through this
// traits struct so the kernel code stays generic (templated on scalar_t).
// -------------------------------------------------------------------------

template<typename T>
struct HalfTraits {
    static __device__ __forceinline__ float to_float(T v);
    static __device__ __forceinline__ T from_float(float v);
};

template<>
struct HalfTraits<__half> {
    static __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
};

template<>
struct HalfTraits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
};

template<>
struct HalfTraits<float> {
    static __device__ __forceinline__ float to_float(float v) { return v; }
    static __device__ __forceinline__ float from_float(float v) { return v; }
};


// -------------------------------------------------------------------------
// Fused asymmetric per-channel activation quant + dequant + outlier bypass
// -------------------------------------------------------------------------

template<typename scalar_t>
__global__ void morpho_act_quant_dequant_kernel(
    const scalar_t* __restrict__ x,          // [M, K] input
    scalar_t* __restrict__       x_dq,       // [M, K] output
    const float* __restrict__    act_scale,  // [K] per-channel scale
    const float* __restrict__    act_zero,   // [K] per-channel zero-point
    const bool* __restrict__     outlier_mask, // [K] boolean outlier map  (nullptr if none)
    const float* __restrict__    comp_limit,   // [K] compensation threshold (nullptr if none)
    int64_t                      total,        // M * K
    int64_t                      K,            // last dimension
    float                        qmax,         // 2^bits - 1
    bool                         has_outliers  // skip bypass logic if false
) {
    using HT = HalfTraits<scalar_t>;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t col = idx % K;
    float val = HT::to_float(x[idx]);

    // ---- Outlier bypass (sparse compensation) ----
    // If this channel is an outlier AND |val| exceeds the compensation
    // threshold, the original value bypasses quantization entirely.
    // The bias compensation term  (x_orig - x_q) * W[:, col]  is
    // implicitly handled by the downstream matmul on the full weight.
    if (has_outliers && outlier_mask[col]) {
        float limit = comp_limit[col];
        if (fabsf(val) > limit) {
            x_dq[idx] = HT::from_float(val);
            return;
        }
    }

    // ---- Asymmetric quant + dequant ----
    // x_q   = clamp(round(x / scale + zero), 0, qmax)
    // x_dq  = (x_q - zero) * scale
    float s = act_scale[col];
    float z = act_zero[col];
    float q = nearbyintf(val / s + z);   // hardware round (faster than roundf)
    q = fminf(fmaxf(q, 0.0f), qmax);
    float dq = (q - z) * s;

    x_dq[idx] = HT::from_float(dq);
}

// -------------------------------------------------------------------------
// Launch wrappers (C-linkage so cpp_sources can call them)
// -------------------------------------------------------------------------

extern "C" {

void morpho_cuda_launch_fp16(
    const void* x, void* x_dq,
    const float* scale, const float* zero,
    const bool* mask, const float* limit,
    int64_t total, int64_t K, float qmax, bool has_outliers,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    morpho_act_quant_dequant_kernel<__half>
        <<<blocks, threads, 0, stream>>>(
            (const __half*)x, (__half*)x_dq,
            scale, zero, mask, limit,
            total, K, qmax, has_outliers);
}

void morpho_cuda_launch_bf16(
    const void* x, void* x_dq,
    const float* scale, const float* zero,
    const bool* mask, const float* limit,
    int64_t total, int64_t K, float qmax, bool has_outliers,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    morpho_act_quant_dequant_kernel<__nv_bfloat16>
        <<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)x, (__nv_bfloat16*)x_dq,
            scale, zero, mask, limit,
            total, K, qmax, has_outliers);
}

void morpho_cuda_launch_fp32(
    const void* x, void* x_dq,
    const float* scale, const float* zero,
    const bool* mask, const float* limit,
    int64_t total, int64_t K, float qmax, bool has_outliers,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    morpho_act_quant_dequant_kernel<float>
        <<<blocks, threads, 0, stream>>>(
            (const float*)x, (float*)x_dq,
            scale, zero, mask, limit,
            total, K, qmax, has_outliers);
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# C++ binding source (compiled with g++)
# ---------------------------------------------------------------------------

_CPP_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Forward declarations (C-linkage, defined in cuda_sources)
extern "C" {
void morpho_cuda_launch_fp16(
    const void* x, void* x_dq,
    const float* scale, const float* zero,
    const bool* mask, const float* limit,
    int64_t total, int64_t K, float qmax, bool has_outliers,
    cudaStream_t stream);

void morpho_cuda_launch_bf16(
    const void* x, void* x_dq,
    const float* scale, const float* zero,
    const bool* mask, const float* limit,
    int64_t total, int64_t K, float qmax, bool has_outliers,
    cudaStream_t stream);

void morpho_cuda_launch_fp32(
    const void* x, void* x_dq,
    const float* scale, const float* zero,
    const bool* mask, const float* limit,
    int64_t total, int64_t K, float qmax, bool has_outliers,
    cudaStream_t stream);
}


torch::Tensor morpho_act_quant_dequant(
    torch::Tensor x,
    torch::Tensor act_scale,
    torch::Tensor act_zero,
    double qmax,
    c10::optional<torch::Tensor> outlier_mask,
    c10::optional<torch::Tensor> comp_limit)
{
    // ---- Input validation ----
    TORCH_CHECK(x.is_cuda(),       "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(act_scale.is_cuda() && act_scale.is_contiguous(),
                "act_scale must be a contiguous CUDA tensor");
    TORCH_CHECK(act_zero.is_cuda() && act_zero.is_contiguous(),
                "act_zero must be a contiguous CUDA tensor");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    // ---- Flatten leading dims → [M, K] ----
    auto orig_shape = x.sizes().vec();
    int64_t K = x.size(-1);
    auto x_flat = x.reshape({-1, K}).contiguous();
    int64_t M = x_flat.size(0);
    int64_t total = M * K;

    auto x_dq = torch::empty_like(x_flat);

    // ---- Determine outlier state ----
    bool has_out = outlier_mask.has_value() && comp_limit.has_value()
                   && outlier_mask->numel() > 0;

    const bool*  mask_ptr  = nullptr;
    const float* limit_ptr = nullptr;
    if (has_out) {
        TORCH_CHECK(outlier_mask->is_contiguous(), "outlier_mask must be contiguous");
        TORCH_CHECK(comp_limit->is_contiguous(),  "comp_limit must be contiguous");
        mask_ptr  = outlier_mask->data_ptr<bool>();
        limit_ptr = comp_limit->data_ptr<float>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

    // ---- Dispatch by dtype ----
    auto dtype = x.scalar_type();
    if (dtype == torch::kHalf) {
        morpho_cuda_launch_fp16(
            x_flat.data_ptr(), x_dq.data_ptr(),
            act_scale.data_ptr<float>(), act_zero.data_ptr<float>(),
            mask_ptr, limit_ptr, total, K, (float)qmax, has_out, stream);
    } else if (dtype == torch::kBFloat16) {
        morpho_cuda_launch_bf16(
            x_flat.data_ptr(), x_dq.data_ptr(),
            act_scale.data_ptr<float>(), act_zero.data_ptr<float>(),
            mask_ptr, limit_ptr, total, K, (float)qmax, has_out, stream);
    } else if (dtype == torch::kFloat32) {
        morpho_cuda_launch_fp32(
            x_flat.data_ptr(), x_dq.data_ptr(),
            act_scale.data_ptr<float>(), act_zero.data_ptr<float>(),
            mask_ptr, limit_ptr, total, K, (float)qmax, has_out, stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", dtype,
                    " (expected half, bfloat16, or float32)");
    }

    return x_dq.reshape(orig_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morpho_act_quant_dequant", &morpho_act_quant_dequant,
          "Fused per-channel asymmetric activation quant+dequant + outlier bypass",
          py::arg("x"),
          py::arg("act_scale"),
          py::arg("act_zero"),
          py::arg("qmax"),
          py::arg("outlier_mask")  = py::none(),
          py::arg("comp_limit")    = py::none());
}
"""


# ---------------------------------------------------------------------------
# JIT compilation (lazy, first-call)
# ---------------------------------------------------------------------------

def _load_cuda_module():
    """Compile and load the CUDA kernel via load_inline.  Cached after first call."""
    global _morpho_cuda_module, _load_attempted

    if _load_attempted:
        return _morpho_cuda_module
    _load_attempted = True

    try:
        from torch.utils.cpp_extension import load_inline

        _morpho_cuda_module = load_inline(
            name="morpho_cuda",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            with_cuda=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        print("[MorphoCUDA] Kernel compiled and loaded successfully.")
        return _morpho_cuda_module

    except Exception as e:
        warnings.warn(
            f"[MorphoCUDA] Failed to compile CUDA kernel: {e}\n"
            f"  Falling back to PyTorch element-wise path. "
            f"Performance will be degraded."
        )
        _morpho_cuda_module = None
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_act_quant_dequant(
    x: torch.Tensor,
    act_scale: torch.Tensor,
    act_zero: torch.Tensor,
    qmax: float,
    outlier_mask: Optional[torch.Tensor] = None,
    comp_limit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused per-channel asymmetric activation quant+dequant with outlier bypass.

    Replaces the following PyTorch ops with a single CUDA kernel::

        x_fp32 = x.float()
        x_scaled = x_fp32 / act_scale + act_zero
        x_int = torch.round(x_scaled).clamp(0, qmax)
        x_dq = (x_int - act_zero) * act_scale
        # optionally: outlier bypass
        x_dq = x_dq.to(x.dtype)

    Args:
        x:              Input activation tensor  ``[..., K]`` in fp16/bf16/fp32.
        act_scale:      Per-channel scale  ``[K]`` fp32.
        act_zero:       Per-channel zero-point  ``[K]`` fp32.
        qmax:           Quantization range max (e.g. 15 for 4-bit).
        outlier_mask:   Boolean mask ``[K]`` marking outlier channels.
                        Pass ``None`` if no outliers.
        comp_limit:     Compensation threshold ``[K]`` fp32.
                        Pass ``None`` if no outliers.

    Returns:
        Dequantized activation ``[..., K]`` in the same dtype as ``x``.
    """
    mod = _load_cuda_module()
    if mod is None:
        return _fallback_pytorch(x, act_scale, act_zero, qmax,
                                 outlier_mask, comp_limit)

    # The C++ function requires contiguous tensors on CUDA
    if not x.is_contiguous():
        x = x.contiguous()
    if not act_scale.is_contiguous():
        act_scale = act_scale.contiguous()
    if not act_zero.is_contiguous():
        act_zero = act_zero.contiguous()

    return mod.morpho_act_quant_dequant(
        x, act_scale, act_zero, qmax, outlier_mask, comp_limit
    )


# ---------------------------------------------------------------------------
# Fallback: pure-PyTorch reference path
# ---------------------------------------------------------------------------

def _fallback_pytorch(
    x: torch.Tensor,
    act_scale: torch.Tensor,
    act_zero: torch.Tensor,
    qmax: float,
    outlier_mask: Optional[torch.Tensor] = None,
    comp_limit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-PyTorch reference that matches the CUDA kernel behaviour exactly."""
    orig_shape = x.shape
    input_dtype = x.dtype
    if x.dim() > 2:
        x = x.reshape(-1, x.size(-1))

    x_fp32 = x.float()

    # Asymmetric quant + dequant
    x_scaled = x_fp32 / act_scale.unsqueeze(0) + act_zero.unsqueeze(0)
    x_int = torch.round(x_scaled).clamp(0, qmax)
    x_dq = (x_int - act_zero.unsqueeze(0)) * act_scale.unsqueeze(0)

    # Outlier bypass (sparse compensation)
    if outlier_mask is not None and comp_limit is not None and outlier_mask.any():
        mask_bool = outlier_mask.to(dtype=torch.bool, device=x.device)
        above_limit = x_fp32.abs() > comp_limit.unsqueeze(0).to(device=x.device)
        bypass = above_limit & mask_bool.unsqueeze(0)
        x_dq = torch.where(bypass, x_fp32, x_dq)

    x_dq = x_dq.to(input_dtype)

    if len(orig_shape) > 2:
        x_dq = x_dq.reshape(orig_shape)
    return x_dq


# ---------------------------------------------------------------------------
# Unit test helpers
# ---------------------------------------------------------------------------

def is_cuda_available() -> bool:
    """Return True if the CUDA kernel was compiled and loaded successfully."""
    return _load_cuda_module() is not None
