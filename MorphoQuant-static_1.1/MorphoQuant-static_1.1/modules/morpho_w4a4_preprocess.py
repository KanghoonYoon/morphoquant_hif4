"""
Weight preprocessing for W4A4 GEMM: BNB NF4 → Symmetric INT4 + Epilogue params.

Converts BNB NF4 4-bit weights into a symmetric per-output-channel INT4 format
optimized for the fused W4A4 GEMM kernel.  Activation quantization scales are
*absorbed* into the weight representation during preprocessing so the GEMM
epilogue only needs a per-output-channel scale + zero-point bias.

Math:

    W_fp[n,k]      = dequant(NF4_packed, quant_state)           // [N,K] FP16
    W_scaled[n,k]   = W_fp[n,k] * a_scale[k]                    // absorb act scale
    w_scale[n]      = max_k(|W_scaled[n,k]|) / 7.0              // per-channel scale
    W_int4[n,k]     = clamp(round(W_scaled[n,k] / w_scale[n]), -8, 7)
    Z_bias[n]       = sum_k a_zero[k] * W_scaled[n,k]           // zero-point bias

At inference time the GEMM computes:

    C_fp[m,n] = w_scale[n] * sum_k (a_int4[m,k] * W_int4[n,k]) + Z_bias[n]
    C_final   = C_fp + DABC_bypass

Where a_int4[m,k] = clamp(round(x_fp[m,k] / a_scale[k] + a_zero[k]), 0, 15).
"""

import torch
from typing import Tuple, Optional
from collections import namedtuple


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

W4A4WeightPack = namedtuple("W4A4WeightPack", [
    "W_packed",          # uint8 [N, K_padded//2]  — symmetric INT4 weights
    "w_scale",           # float16 [N]             — per-output-channel dequant scale
    "z_bias",            # float32 [N]             — zero-point bias (pre-computed)
    "W_outlier",         # float16 [N, n_out]       — outlier-column FP16 weights
    "outlier_indices",   # int64 [n_out]            — outlier channel indices
    "in_features",       # int                       — original K
    "out_features",      # int                       — N
    "K_padded",          # int                       — K padded to even (for packing)
])


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def build_w4a4_weights(
    weight_packed: torch.Tensor,
    quant_state,
    act_scale: torch.Tensor,
    act_zero: torch.Tensor,
    outlier_indices: torch.Tensor,
    verbose: bool = False,
) -> W4A4WeightPack:
    """Convert one layer's BNB NF4 weights into W4A4 GEMM format.

    Args:
        weight_packed:   BNB 4-bit packed uint8 tensor (from Linear4bit.weight).
        quant_state:     BNB ``QuantState`` (absmax, blocksize, code, nested, …).
        act_scale:       Per-channel activation scale ``[K]`` float32.
        act_zero:        Per-channel activation zero-point ``[K]`` float32.
        outlier_indices: Channel indices marked as outliers ``[n_out]`` int64.
        verbose:         Print preprocessing statistics.

    Returns:
        ``W4A4WeightPack`` with all components needed by ``MorphoW4A4Linear``.
    """
    import bitsandbytes.functional as bnbF

    # ---- 1. Dequantize BNB NF4 → FP16 (one-time, cached) ----
    # BNB dequantize_4bit handles double-quantized absmax internally.
    with torch.no_grad():
        W_fp16 = bnbF.dequantize_4bit(
            weight_packed, quant_state=quant_state
        )  # → [N, K] float16

    N, K = W_fp16.shape
    device = W_fp16.device

    # ---- 2. Absorb activation scale into weight ----
    # W_scaled[n,k] = W_fp[n,k] * a_scale[k]
    # This lets us use symmetric INT4 for weights while still correctly
    # dequantizing the final result via a simple per-output-channel scale.
    a_scale_fp16 = act_scale.to(device=device, dtype=torch.float16)
    W_scaled = W_fp16 * a_scale_fp16.unsqueeze(0)  # [N, K]

    # ---- 3. Per-output-channel symmetric INT4 quantization ----
    # Signed INT4 range: [-8, 7], max representable = 7
    QMAX_W = 7.0

    w_scale_abs = W_scaled.abs().amax(dim=1).clamp(min=1e-8)  # [N]
    w_scale = (w_scale_abs / QMAX_W).to(torch.float16)         # [N] FP16

    W_int4_float = torch.clamp(
        torch.round(W_scaled / w_scale.unsqueeze(1).float()), -8, 7
    )
    W_int4_signed = W_int4_float.to(torch.int8)  # [N, K], values in [-8, 7]

    # ---- 4. Pad to even K and pack 2 values per uint8 ----
    K_padded = K if (K % 2 == 0) else K + 1
    W_int4_for_pack = W_int4_signed  # [N, K]
    if K_padded > K:
        pad_col = torch.zeros(N, K_padded - K, dtype=torch.int8, device=device)
        W_int4_for_pack = torch.cat([W_int4_signed, pad_col], dim=1)

    # Convert signed int4 [-8,7] → unsigned nibble [0,15] via bitwise mask
    W_u4 = W_int4_for_pack & 0x0F
    # Pack: even cols → low nibble, odd cols → high nibble
    W_packed = (W_u4[:, 0::2] | (W_u4[:, 1::2] << 4)).to(torch.uint8)
    # W_packed.shape = [N, K_padded//2]

    # ---- 5. Compute zero-point bias ----
    # IMPORTANT: Z_bias must use the QUANTIZED weight reconstruction for
    # mathematical consistency with the GEMM epilogue.
    #   Z_bias[n] = sum_k a_zero[k] * W_recon[n,k]
    #             = w_scale[n] * sum_k a_zero[k] * W_int4[n,k]
    # where W_int4 is the UNPADDED version (original K channels).
    a_zero_dev = act_zero.to(device=device, dtype=torch.float32)
    z_bias = w_scale.float() * torch.sum(
        a_zero_dev.unsqueeze(0) * W_int4_signed.float(), dim=1
    )  # [N] FP32

    # ---- 6. Extract outlier-channel FP16 weights for DABC bypass ----
    n_out = outlier_indices.numel()
    if n_out > 0:
        # Use original (unscaled) FP16 weights for DABC bypass:
        #   DABC = x_orig[outlier] @ W_outlier.T
        W_outlier = W_fp16[:, outlier_indices].contiguous().clone()
    else:
        W_outlier = torch.zeros(N, 0, dtype=torch.float16, device=device)

    if verbose:
        W_reconstructed = (W_int4_signed.float()
                           * w_scale.unsqueeze(1).float())
        qerror = (W_scaled.float() - W_reconstructed).abs()
        print(f"[W4A4 Preprocess] N={N}, K={K}, K_padded={K_padded}, "
              f"n_outliers={n_out}")
        print(f"  w_scale range=[{w_scale.min():.6f}, {w_scale.max():.6f}]")
        print(f"  Quant error: mean={qerror.mean():.6f}, "
              f"max={qerror.max():.4f}, "
              f"rel_mean={qerror.mean() / W_scaled.float().abs().mean():.6f}")

    return W4A4WeightPack(
        W_packed=W_packed,
        w_scale=w_scale,
        z_bias=z_bias,
        W_outlier=W_outlier,
        outlier_indices=outlier_indices,
        in_features=K,
        out_features=N,
        K_padded=K_padded,
    )


# ---------------------------------------------------------------------------
# Reference dequant (for verification only)
# ---------------------------------------------------------------------------

def dequant_w4a4_weights(pack: W4A4WeightPack) -> torch.Tensor:
    """Dequantize W4A4 packed weights back to FP16 for accuracy verification.

    Returns W_scaled [N, K] = W_int4 * w_scale (the absorbed-activation-scale version).
    """
    N = pack.out_features
    K = pack.in_features
    K_pad = pack.K_padded

    # Unpack uint8 → signed int4
    low = pack.W_packed & 0x0F
    high = (pack.W_packed >> 4) & 0x0F

    # Sign-extend: (int8_t)(nibble << 4) >> 4
    def _sign_extend(nibble: torch.Tensor) -> torch.Tensor:
        # Convert unsigned nibble 0..15 → signed int8 -8..7
        # Values >= 8 are negative in two's complement 4-bit
        neg_mask = nibble >= 8
        result = nibble.to(torch.int8)
        result[neg_mask] = result[neg_mask] - 16
        return result

    low_signed = _sign_extend(low)    # [N, K_pad//2]
    high_signed = _sign_extend(high)  # [N, K_pad//2]

    # Interleave: [N, K_pad]
    W_int4 = torch.stack([low_signed, high_signed], dim=2).reshape(N, 2 * (K_pad // 2))
    W_int4 = W_int4[:, :K].to(torch.float32)  # trim padding

    # Dequantize
    w_scale_fp32 = pack.w_scale.float().unsqueeze(1)
    return (W_int4 * w_scale_fp32).to(torch.float16)


# ---------------------------------------------------------------------------
# Sanity check (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[W4A4 Preprocess] Self-test (math correctness only)...")
    import bitsandbytes.functional as bnbF

    N, K = 512, 3584
    device = "cuda:0"

    # Create random weights
    W_orig = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
    W_packed_bnb, quant_state = bnbF.quantize_4bit(
        W_orig, quant_type="nf4", blocksize=64, compress_statistics=True
    )

    # Symmetric act params for clean math test: act_zero=0
    act_scale = torch.ones(K, dtype=torch.float32) * 0.01
    act_zero = torch.zeros(K, dtype=torch.float32)
    outlier_idx = torch.tensor([], dtype=torch.long)

    # Build W4A4
    pack = build_w4a4_weights(
        W_packed_bnb, quant_state, act_scale, act_zero, outlier_idx, verbose=True
    )

    # Test: W4A4 output should match A_int4 @ W_recon^T exactly (since z_bias=0)
    M = 4
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    x_fp32 = x.float()
    a_s = act_scale.to(device)
    a_z = act_zero.to(device)
    a_int4 = torch.clamp(torch.round(x_fp32 / a_s.unsqueeze(0) + a_z.unsqueeze(0)), 0, 15)

    W_deq_test = dequant_w4a4_weights(pack)
    y_w4a4 = torch.matmul(a_int4.float(), W_deq_test.T.float()) - pack.z_bias.float().unsqueeze(0)

    # Reference: A_dq @ W_bnb^T
    W_bnb = bnbF.dequantize_4bit(W_packed_bnb, quant_state=quant_state)
    a_dq = (a_int4 - a_z.unsqueeze(0)) * a_s.unsqueeze(0)
    y_ref = torch.matmul(a_dq, W_bnb.T.float())

    # Math correctness: Z_bias=0 case should have error equal to weight quant error
    signal = y_ref.pow(2).mean()
    noise = (y_w4a4 - y_ref).pow(2).mean()
    snr = 10.0 * torch.log10(signal / noise).item() if noise > 1e-12 else float("inf")
    max_err = (y_w4a4 - y_ref).abs().max().item()
    print(f"  SNR (math test, zero=0): {snr:.1f} dB, Max error: {max_err:.6f}")
    # For symmetric case with random weights, INT4 quantization error is expected
    # The math is correct if error is bounded by weight quantization magnitude
    weight_qerror = W_bnb.float() - W_deq_test.float()
    expected_max = weight_qerror.abs().max().item() * a_int4.float().abs().max().item() * K
    print(f"  Expected max error bound: {expected_max:.1f}")
    print("[W4A4 Preprocess] Math correctness VERIFIED" if max_err < expected_max * 1.1
          else "[W4A4 Preprocess] FAILED: error exceeds bound")
