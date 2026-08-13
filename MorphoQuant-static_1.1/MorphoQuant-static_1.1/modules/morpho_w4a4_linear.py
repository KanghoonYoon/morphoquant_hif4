"""
MorphoW4A4Linear — W4A4 quantized linear layer with DABC epilogue.

Stores weights in packed symmetric INT4 format (~0.5 bytes/element) with
per-output-channel dequantization scales.  Activation quantization is fused
into the GEMM kernel's load path, and channel-wise DABC (bypass bias
compensation for outlier channels) is applied via a separate FP16 matmul.

Compared to MorphoFusedLinear (14.40 GB W_fp16_cache):
    W4A4 stores weights at ~3.60 GB packed INT4 → **-75% weight memory**.

Forward path::

    x → [fused act_quant + W4A4 GEMM] → y_main
    if outliers: y_main += x_outlier @ W_outlier.T   (DABC bypass)
    return y
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import namedtuple

# Preprocessing result type (mirrors morpho_w4a4_preprocess.W4A4WeightPack)
W4A4WeightPack = namedtuple("W4A4WeightPack", [
    "W_packed", "w_scale", "z_bias",
    "W_outlier", "outlier_indices",
    "in_features", "out_features", "K_padded",
])


# ---------------------------------------------------------------------------
# MorphoW4A4Linear
# ---------------------------------------------------------------------------

class MorphoW4A4Linear(nn.Module):
    """W4A4 quantized linear layer with DABC outlier bypass.

    Weights are stored in packed symmetric INT4 format with per-output-channel
    dequantization scales.  Activation quantization is fused into the GEMM
    kernel.  DABC (channel-wise outlier bias compensation) is applied via a
    separate lightweight FP16 matmul for outlier channels.

    Memory::

        W_packed   : uint8  [N, K/2]     (~0.5 bytes per weight element)
        w_scale    : float16 [N]         (2 bytes per output channel)
        z_bias     : float32 [N]         (4 bytes per output channel)
        W_outlier  : float16 [N, n_out]  (2 bytes per outlier weight element)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        W_packed: torch.Tensor,           # [N, K_padded/2] uint8
        w_scale: torch.Tensor,            # [N] float16
        z_bias: torch.Tensor,             # [N] float32
        act_scale: torch.Tensor,          # [K] float32
        act_zero: torch.Tensor,           # [K] float32
        W_outlier: Optional[torch.Tensor] = None,     # [N, n_out] float16
        outlier_indices: Optional[torch.Tensor] = None,  # [n_out] int64
        compensation_limit: Optional[torch.Tensor] = None,  # [K] float32
        activation_bit: int = 4,
        K_padded: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bit = activation_bit
        self.qmax = float(2 ** activation_bit - 1)

        # ---- Weight storage (packed INT4) ----
        self.register_buffer("W_packed", W_packed.to(torch.uint8))

        # ---- Epilogue parameters ----
        self.register_buffer("w_scale", w_scale.to(torch.float16))
        self.register_buffer("z_bias", z_bias.to(torch.float32))
        self.register_buffer("act_scale", act_scale.to(torch.float32))
        self.register_buffer("act_zero", act_zero.to(torch.float32))

        # ---- K_padded (may differ from in_features for alignment) ----
        self.K_padded = K_padded if K_padded is not None else in_features

        # ---- DABC (outlier bypass) ----
        n_out = outlier_indices.numel() if outlier_indices is not None else 0
        self.has_outliers = n_out > 0
        if self.has_outliers:
            self.register_buffer("outlier_indices", outlier_indices.to(torch.long))
            self.register_buffer("W_outlier", W_outlier.to(torch.float16))
            if compensation_limit is not None:
                self.register_buffer("compensation_limit",
                                     compensation_limit.to(torch.float32))
        else:
            # Always register for TorchScript / state_dict consistency
            self.register_buffer(
                "outlier_indices", torch.zeros(0, dtype=torch.long)
            )
            self.register_buffer(
                "W_outlier", torch.zeros(out_features, 0, dtype=torch.float16)
            )

        # ---- CUDA kernel availability ----
        self._use_cuda = self._init_cuda()

    def _init_cuda(self) -> bool:
        """Try to load the W4A4 CUDA kernel."""
        try:
            from modules.morpho_w4a4_gemm_kernel import is_cuda_available
            return is_cuda_available()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused W4A4 forward.

        Args:
            x: Input activation ``[..., K]`` half/bfloat16.

        Returns:
            Output ``[..., N]`` in the same dtype as input.
        """
        orig_shape = x.shape
        input_dtype = x.dtype

        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)

        # ---- Main W4A4 GEMM (act quant fused in kernel) ----
        if self._use_cuda:
            from modules.morpho_w4a4_gemm_kernel import w4a4_gemm_forward

            y = w4a4_gemm_forward(
                x,
                self.W_packed,
                self.act_scale,
                self.act_zero,
                self.w_scale,
                self.z_bias,
                self.qmax,
            )
        else:
            # Fallback: pure PyTorch path
            x_fp32 = x.float()
            a_s = self.act_scale
            a_z = self.act_zero

            # Act quantize → (A_int4 - a_zero), no a_scale (absorbed in W)
            a_int = torch.clamp(
                torch.round(x_fp32 / a_s + a_z), 0, self.qmax
            )
            a_centered = (a_int - a_z).float()

            # Unpack + dequant weights
            low = (self.W_packed & 0x0F).to(torch.int8)
            high = ((self.W_packed >> 4) & 0x0F).to(torch.int8)
            low_s = torch.where(low >= 8, low - 16, low).float()
            high_s = torch.where(high >= 8, high - 16, high).float()
            W_int4 = torch.stack([low_s, high_s], dim=2).reshape(
                self.out_features, -1
            )[:, :self.in_features]

            # W4A4 matmul: C = (A_int4 - a_zero) @ (W_int4 * w_scale)^T
            W_deq = W_int4 * self.w_scale.float().unsqueeze(1)
            y = a_centered @ W_deq.T
            y = y.to(dtype=input_dtype)

        # ---- DABC: outlier bypass compensation ----
        if self.has_outliers and self.outlier_indices.numel() > 0:
            # Extract original (non-quantized) values at outlier channels
            x_outlier = torch.index_select(
                x, dim=-1, index=self.outlier_indices.to(device=x.device)
            )
            # DABC = x_outlier @ W_outlier^T  (small FP16 matmul via cuBLAS)
            y_dabc = torch.matmul(
                x_outlier.to(self.W_outlier.dtype), self.W_outlier.T
            ).to(dtype=y.dtype)
            y = y + y_dabc

        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], self.out_features)
        return y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def compute_snr(self, x_fp16: torch.Tensor, W_fp16_ref: torch.Tensor) -> float:
        """Compute SNR (dB) between W4A4 output and FP16 reference.

        Args:
            x_fp16: Input activation.
            W_fp16_ref: Reference FP16 weight [out_features, in_features].

        Returns:
            SNR in dB.
        """
        with torch.no_grad():
            y_w4a4 = self.forward(x_fp16)
            y_ref = torch.matmul(
                x_fp16.to(W_fp16_ref.device).float(),
                W_fp16_ref.T.float()
            )
            signal = y_ref.float().pow(2).mean()
            noise = (y_w4a4.float() - y_ref.float()).pow(2).mean()
            if noise < 1e-12:
                return float("inf")
            return 10.0 * torch.log10(signal / noise).item()

    def extra_repr(self) -> str:
        n_out = self.outlier_indices.numel() if self.outlier_indices is not None else 0
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bit={self.activation_bit}, outliers={n_out}, "
                f"cuda={self._use_cuda}")


# ---------------------------------------------------------------------------
# Model replacement
# ---------------------------------------------------------------------------

def replace_morpho_with_w4a4(
    model: nn.Module,
    layer_params: Optional[Dict[str, dict]] = None,
    activation_bit: int = 4,
    verbose: bool = True,
) -> nn.Module:
    """Replace all MorphoQuant quantized layers with MorphoW4A4Linear.

    Handles both ``MorphoQuantActWrapper`` and ``Linear4bit + QuantAct`` patterns.
    Uses W4A4 INT4 weight preprocessing to convert BNB NF4 → symmetric INT4.

    Args:
        model:           Calibrated MorphoQuant model.
        layer_params:    Pre-extracted params (if None, calls extract_morpho_params).
        activation_bit:  Activation bit-width.
        verbose:         Print replacement progress.

    Returns:
        The same model with quantized layers → MorphoW4A4Linear.
    """
    from bitsandbytes.nn.modules import Linear4bit
    from modules.model_factory import MorphoQuantActWrapper
    from modules.morpho_fused_linear import extract_morpho_params, build_fused_weights
    from modules.morpho_w4a4_preprocess import build_w4a4_weights

    if layer_params is None:
        layer_params = extract_morpho_params(model)

    if not layer_params:
        print("[W4A4] replace_morpho_with_w4a4: no quantized layers found.")
        return model

    replaced = 0
    skipped = 0

    for name, params in layer_params.items():
        try:
            # First dequant BNB to get FP16 weights and act params
            act_scale, act_zero, W_fp16_cache, W_outlier_fp16 = build_fused_weights(
                params, activation_bit=activation_bit
            )

            # Now build W4A4 packed weights from the dequantized FP16
            import bitsandbytes.functional as bnbF
            weight_packed = params.get("weight_packed")
            quant_state = params.get("weight_quant_state")

            if weight_packed is None or quant_state is None:
                if verbose:
                    print(f"  [SKIP] {name}: missing BNB weight data")
                skipped += 1
                continue

            pack = build_w4a4_weights(
                weight_packed=weight_packed,
                quant_state=quant_state,
                act_scale=act_scale,
                act_zero=act_zero,
                outlier_indices=params["outlier_indices"],
                verbose=False,
            )

        except Exception as e:
            if verbose:
                print(f"  [SKIP] {name}: weight build failed ({e})")
            skipped += 1
            continue

        w4a4_linear = MorphoW4A4Linear(
            in_features=pack.in_features,
            out_features=pack.out_features,
            W_packed=pack.W_packed,
            w_scale=pack.w_scale,
            z_bias=pack.z_bias,
            act_scale=act_scale,
            act_zero=act_zero,
            W_outlier=pack.W_outlier,
            outlier_indices=pack.outlier_indices,
            compensation_limit=params.get("compensation_limit"),
            activation_bit=activation_bit,
            K_padded=pack.K_padded,
        )

        # Navigate to parent module and replace
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        existing = getattr(parent, parts[-1])
        if isinstance(existing, (MorphoQuantActWrapper, Linear4bit)):
            setattr(parent, parts[-1], w4a4_linear)
            replaced += 1
        else:
            skipped += 1

    if verbose:
        print(f"[MorphoW4A4] Replaced {replaced} layers, skipped {skipped}")

    return model
