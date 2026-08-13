"""
JIT-compiled W4A4 GEMM CUDA extension for MorphoQuant.

Provides ``w4a4_gemm_forward()`` — a fused 4-bit weight × 4-bit activation GEMM
with epilogue (weight scale + zero-point bias).  The activation quantization step
is fused into the GEMM kernel's load path, eliminating the separate act_quant
kernel launch.

Uses ``torch.utils.cpp_extension.load_inline`` for JIT compilation at import
time.  Falls back gracefully if CUDA toolkit is unavailable.
"""

import torch
import warnings
from typing import Optional

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_w4a4_cuda_module = None
_load_attempted = False


# ---------------------------------------------------------------------------
# CUDA kernel source (compiled with nvcc)
# ---------------------------------------------------------------------------

# Imported from morpho_w4a4_gemm.cu at build time via load_inline.
# The raw source is read from the .cu file to keep it as a single point of truth.
import os
_CUDA_SRC_PATH = os.path.join(os.path.dirname(__file__), "morpho_w4a4_gemm.cu")
with open(_CUDA_SRC_PATH, "r") as _f:
    _CUDA_SRC = _f.read()


# ---------------------------------------------------------------------------
# C++ binding source (compiled with g++)
# ---------------------------------------------------------------------------

_CPP_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Forward declarations (C-linkage, defined in cuda_sources)
extern "C" {
void w4a4_gemm_launch_fp16(
    const void* x, const void* W_packed,
    const float* act_scale, const float* act_zero,
    const void* w_scale, const float* z_bias,
    void* out,
    int M, int N, int K, float qmax,
    cudaStream_t stream);

void w4a4_gemm_launch_bf16(
    const void* x, const void* W_packed,
    const float* act_scale, const float* act_zero,
    const void* w_scale, const float* z_bias,
    void* out,
    int M, int N, int K, float qmax,
    cudaStream_t stream);
}


torch::Tensor w4a4_gemm_forward(
    torch::Tensor x,            // [..., K] half/bfloat16
    torch::Tensor W_packed,     // [N, K/2] uint8 packed INT4
    torch::Tensor act_scale,    // [K] float32
    torch::Tensor act_zero,     // [K] float32
    torch::Tensor w_scale,      // [N] float16
    torch::Tensor z_bias,       // [N] float32
    double qmax_d)
{
    // ---- Input validation ----
    TORCH_CHECK(x.is_cuda(),        "x must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(act_scale.is_cuda() && act_scale.is_contiguous(),
                "act_scale must be a contiguous CUDA tensor");
    TORCH_CHECK(act_zero.is_cuda() && act_zero.is_contiguous(),
                "act_zero must be a contiguous CUDA tensor");
    TORCH_CHECK(w_scale.is_cuda(),  "w_scale must be a CUDA tensor");
    TORCH_CHECK(z_bias.is_cuda(),   "z_bias must be a CUDA tensor");
    TORCH_CHECK(W_packed.dtype() == torch::kUInt8,
                "W_packed must be uint8");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    auto dtype = x.scalar_type();
    TORCH_CHECK(dtype == torch::kHalf || dtype == torch::kBFloat16,
                "x must be half or bfloat16, got ", dtype);

    // ---- Flatten leading dims → [M, K] ----
    auto orig_shape = x.sizes().vec();
    int64_t K = x.size(-1);
    auto x_flat = x.reshape({-1, K}).contiguous();
    int64_t M = x_flat.size(0);
    int64_t N = W_packed.size(0);

    TORCH_CHECK(act_scale.size(0) == K,
                "act_scale size ", act_scale.size(0), " != K=", K);
    TORCH_CHECK(w_scale.size(0) == N,
                "w_scale size ", w_scale.size(0), " != N=", N);

    // ---- Ensure W_packed is contiguous ----
    auto W_contig = W_packed.contiguous();

    // ---- Allocate output ----
    auto out = torch::empty({M, N}, x.options());

    float qmax = (float)qmax_d;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

    // ---- Dispatch by dtype ----
    if (dtype == torch::kHalf) {
        w4a4_gemm_launch_fp16(
            x_flat.data_ptr(), W_contig.data_ptr(),
            act_scale.data_ptr<float>(), act_zero.data_ptr<float>(),
            w_scale.data_ptr(), z_bias.data_ptr<float>(),
            out.data_ptr(),
            (int)M, (int)N, (int)K, qmax, stream);
    } else {
        w4a4_gemm_launch_bf16(
            x_flat.data_ptr(), W_contig.data_ptr(),
            act_scale.data_ptr<float>(), act_zero.data_ptr<float>(),
            w_scale.data_ptr(), z_bias.data_ptr<float>(),
            out.data_ptr(),
            (int)M, (int)N, (int)K, qmax, stream);
    }

    // ---- Reshape output to match input batch dims ----
    auto out_shape = orig_shape;
    out_shape[out_shape.size() - 1] = N;
    return out.reshape(out_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a4_gemm_forward", &w4a4_gemm_forward,
          "W4A4 GEMM: fused act quant + INT4 weight GEMM + epilogue (w_scale, Z_bias)",
          py::arg("x"),
          py::arg("W_packed"),
          py::arg("act_scale"),
          py::arg("act_zero"),
          py::arg("w_scale"),
          py::arg("z_bias"),
          py::arg("qmax") = 15.0);
}
"""


# ---------------------------------------------------------------------------
# JIT compilation (lazy, first-call)
# ---------------------------------------------------------------------------

def _load_cuda_module():
    """Compile and load the W4A4 CUDA kernel via load_inline.  Cached after first call."""
    global _w4a4_cuda_module, _load_attempted

    if _load_attempted:
        return _w4a4_cuda_module
    _load_attempted = True

    try:
        from torch.utils.cpp_extension import load_inline

        _w4a4_cuda_module = load_inline(
            name="morpho_w4a4_gemm",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            with_cuda=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        print("[MorphoW4A4] CUDA GEMM kernel compiled and loaded successfully.")
        return _w4a4_cuda_module

    except Exception as e:
        warnings.warn(
            f"[MorphoW4A4] Failed to compile CUDA kernel: {e}\n"
            f"  Falling back to PyTorch reference path. "
            f"Performance will be degraded."
        )
        _w4a4_cuda_module = None
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def w4a4_gemm_forward(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    act_scale: torch.Tensor,
    act_zero: torch.Tensor,
    w_scale: torch.Tensor,
    z_bias: torch.Tensor,
    qmax: float = 15.0,
) -> torch.Tensor:
    """Fused W4A4 GEMM: act quant + INT4 weight GEMM + epilogue.

    Computes::

        A_int4[m,k] = clamp(round(x[m,k] / act_scale[k] + act_zero[k]), 0, qmax)
        S[m,n]      = sum_k A_int4[m,k] * W_int4[n,k]
        out[m,n]    = w_scale[n] * S[m,n] - z_bias[n]

    All steps are fused into a single CUDA kernel launch.

    Args:
        x:          Input activation  ``[..., K]``  half or bfloat16.
        W_packed:   Packed INT4 weights ``[N, K/2]``  uint8.
        act_scale:  Per-channel act scale ``[K]`` float32.
        act_zero:   Per-channel act zero-point ``[K]`` float32.
        w_scale:    Per-output weight scale ``[N]`` float16.
        z_bias:     Zero-point bias ``[N]`` float32.
        qmax:       Quantization max (15 for 4-bit).

    Returns:
        Output tensor ``[..., N]`` in the same dtype as ``x``.
    """
    mod = _load_cuda_module()
    if mod is None:
        return _w4a4_fallback_pytorch(
            x, W_packed, act_scale, act_zero, w_scale, z_bias, qmax
        )

    # Ensure contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not W_packed.is_contiguous():
        W_packed = W_packed.contiguous()
    if not act_scale.is_contiguous():
        act_scale = act_scale.contiguous()
    if not act_zero.is_contiguous():
        act_zero = act_zero.contiguous()

    return mod.w4a4_gemm_forward(
        x, W_packed, act_scale, act_zero, w_scale, z_bias, qmax
    )


# ---------------------------------------------------------------------------
# Fallback: pure-PyTorch reference path (for debugging / no-CUDA environments)
# ---------------------------------------------------------------------------

def _w4a4_fallback_pytorch(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    act_scale: torch.Tensor,
    act_zero: torch.Tensor,
    w_scale: torch.Tensor,
    z_bias: torch.Tensor,
    qmax: float = 15.0,
) -> torch.Tensor:
    """Pure-PyTorch reference that matches the CUDA kernel behaviour exactly."""
    orig_shape = x.shape
    input_dtype = x.dtype

    if x.dim() > 2:
        x = x.reshape(-1, x.size(-1))
    M, K = x.shape
    N = W_packed.size(0)

    device = x.device

    # 1. Act quantize: A_int4 = clamp(round(x/scale + zero), 0, qmax)
    x_fp32 = x.float()
    a_s = act_scale.to(device=device)
    a_z = act_zero.to(device=device)
    a_int4 = torch.clamp(
        torch.round(x_fp32 / a_s.unsqueeze(0) + a_z.unsqueeze(0)), 0, qmax
    )  # [M, K]

    # 2. Unpack INT4 weights and dequantize
    low = (W_packed & 0x0F).to(torch.int8)
    high = ((W_packed >> 4) & 0x0F).to(torch.int8)

    # Sign-extend
    low_s = torch.where(low >= 8, low - 16, low).to(torch.float32)
    high_s = torch.where(high >= 8, high - 16, high).to(torch.float32)

    # Interleave: [N, K]
    W_int4 = torch.stack([low_s, high_s], dim=2).reshape(N, -1)[:, :K]

    # 3. W4A4 GEMM: C = (A_int4 - a_zero) @ (W_int4 * w_scale)^T
    # a_scale is absorbed in W_int4 during preprocessing, so no need for it here.
    a_centered = (a_int4 - a_z.unsqueeze(0)).float()
    W_deq = W_int4 * w_scale.to(device=device).float().unsqueeze(1)
    out = torch.matmul(a_centered, W_deq.transpose(1, 0))  # [M, N]

    out = out.to(dtype=input_dtype)

    if len(orig_shape) > 2:
        out = out.reshape(*orig_shape[:-1], N)
    return out


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def is_cuda_available() -> bool:
    """Return True if the W4A4 CUDA kernel was compiled and loaded successfully."""
    return _load_cuda_module() is not None
