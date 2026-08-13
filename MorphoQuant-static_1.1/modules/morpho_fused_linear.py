"""
MorphoQuant Fused Linear Layer with Cached FP16 Weights.

Replaces the QuantAct (activation quantization) + BNB Linear4bit (weight
dequant on every forward) + F.linear combination with a single fused layer:

    Act quant+dequant (per-channel asymmetric) → sparse compensation
    → F.linear(x_dq, W_fp16_cache)

The BNB 4-bit weight is dequantized to FP16 **once during build** and cached,
eliminating per-forward-pass dequant overhead. Activation quantization uses
per-channel parameters (best_min/best_max) extracted from a real calibrated
MorphoQuant model.

Note: The INT8 TensorCore path (via ``torch._int_mm``) was explored but found
to be consistently slower than cached FP16 matmul on NVIDIA L20 for the layer
dimensions in Qwen2.5-Omni-7B.  It is kept for reference in git history but
removed from the active code path.

Reference: test_morpho_cuda_final2.py (Strategy 2: Channel Extract Bypass)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Parameter extraction from a calibrated MorphoQuant model
# ---------------------------------------------------------------------------

def extract_morpho_params(model: nn.Module) -> Dict[str, dict]:
    """Traverse a calibrated MorphoQuant model and extract per-layer fusion parameters.

    Handles both layer patterns used by the MorphoQuant build:
    1. ``MorphoQuantActWrapper`` (used by _wrap_morpho_quantact_recursive)
    2. ``Linear4bit`` with ``.quant_activation`` attribute (used by standard morpho path)

    For each quantized layer found, collects:

    * ``outlier_indices`` -- tensor of channel indices marked as outliers
      (from ``QuantAct.outlier_mask``)
    * ``best_min`` / ``best_max`` -- per-channel asymmetric quantization range
      (from ``QuantAct.activation_range_min/max``)
    * ``compensation_limit`` -- per-channel threshold above which sparse
      compensation bypasses quantization
    * ``weight_quant_state`` -- BNB QuantState for 4-bit weight dequantization
    * ``in_features`` / ``out_features`` -- layer dimensions

    Returns:
        Dict mapping layer name → params dict.
    """
    from bitsandbytes.nn.modules import Linear4bit
    from modules.model_factory import MorphoQuantActWrapper

    params = {}
    for name, module in model.named_modules():

        # Pattern 1: MorphoQuantActWrapper (some build paths)
        if isinstance(module, MorphoQuantActWrapper):
            qact = module.quant_activation
            inner = module.module
            in_f = inner.in_features
            out_f = inner.out_features
            quant_state = getattr(inner, 'quant_state', None)

        # Pattern 2: Linear4bit with .quant_activation (standard morpho path)
        elif isinstance(module, Linear4bit):
            qact = getattr(module, 'quant_activation', None)
            if qact is None:
                continue
            inner = module
            in_f = inner.in_features
            out_f = inner.out_features
            quant_state = getattr(inner, 'quant_state', None)

        else:
            continue

        if quant_state is None:
            continue

        # 1. Outlier channel indices
        outlier_mask = getattr(qact, 'outlier_mask', None)
        if outlier_mask is not None and outlier_mask.any():
            outlier_indices = outlier_mask.nonzero(as_tuple=True)[0].cpu()
        else:
            outlier_indices = torch.tensor([], dtype=torch.long)

        # 2. Per-channel quantization range
        best_min = getattr(qact, 'activation_range_min', None)
        best_max = getattr(qact, 'activation_range_max', None)
        if best_min is None:
            best_min = getattr(qact, 'llama_range_min', None)
        if best_max is None:
            best_max = getattr(qact, 'llama_range_max', None)

        # 3. Compensation threshold
        compensation_limit = getattr(qact, 'compensation_limit', None)

        params[name] = {
            'outlier_indices': outlier_indices,
            'best_min': best_min.clone() if best_min is not None else None,
            'best_max': best_max.clone() if best_max is not None else None,
            'compensation_limit': compensation_limit.clone() if compensation_limit is not None else None,
            'weight_quant_state': quant_state,
            'weight_packed': inner.weight.data.clone(),  # packed 4-bit tensor
            'in_features': in_f,
            'out_features': out_f,
        }
    return params


# ---------------------------------------------------------------------------
# Weight conversion: BNB 4-bit → INT8 main + FP16 outlier bypass
# ---------------------------------------------------------------------------

def build_fused_weights(
    layer_params: dict,
    activation_bit: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert one layer's BNB 4-bit weights into fused-kernel-ready components.

    Dequantizes the BNB 4-bit weight once and caches it in FP16, eliminating
    per-forward-pass dequant overhead.

    Args:
        layer_params: Output of ``extract_morpho_params()`` for one layer.
        activation_bit: Activation quantization bit-width (default 4).

    Returns:
        Tuple of:
        * ``act_scale``       [in_features]                per-channel act scale
        * ``act_zero``        [in_features]                per-channel act zero-point
        * ``W_fp16_cache``    [out_features, in_features]  FP16 full-weight cache
        * ``W_outlier_fp16``  [out_features, n_out] or None  outlier-column weights
    """
    outlier_idx = layer_params['outlier_indices']       # [num_outliers]
    quant_state = layer_params['weight_quant_state']
    weight_packed = layer_params.get('weight_packed')   # packed 4-bit tensor
    in_features = layer_params['in_features']
    out_features = layer_params['out_features']
    best_min = layer_params.get('best_min')
    best_max = layer_params.get('best_max')

    # ---- 1. Dequantize BNB 4-bit → FP16 (once, cached) ----
    import bitsandbytes.functional as bnbF
    if weight_packed is None:
        raise ValueError("weight_packed is None — extract_morpho_params must be called first")
    if quant_state is None:
        raise ValueError("quant_state is None")
    W_fp16 = bnbF.dequantize_4bit(weight_packed, quant_state=quant_state)  # [N, K]
    W_fp16 = W_fp16.to(torch.float16)

    # ---- 2. Extract outlier-column weights (for SNR / reference) ----
    if outlier_idx.numel() > 0:
        W_outlier_fp16 = W_fp16[:, outlier_idx].contiguous().clone()
    else:
        W_outlier_fp16 = None

    # ---- 3. Activation quantization parameters (per-channel asymmetric) ----
    if best_min is not None and best_max is not None:
        qmax = float(2 ** activation_bit - 1)
        act_scale = (best_max - best_min).clamp(min=1e-8) / qmax
        act_zero = torch.round(-best_min / act_scale).clamp(0, qmax)
    else:
        # Fallback: symmetric per-channel
        act_scale = torch.ones(in_features, dtype=torch.float32)
        act_zero = torch.zeros(in_features, dtype=torch.float32)

    return act_scale, act_zero, W_fp16, W_outlier_fp16


# ---------------------------------------------------------------------------
# MorphoFusedLinear — the fused layer
# ---------------------------------------------------------------------------

class MorphoFusedLinear(nn.Module):
    """Fused MorphoQuant linear layer with cached FP16 weights.

    Replaces the BNB Linear4bit + QuantAct pair with a single fused layer that
    performs per-channel asymmetric activation quantization → sparse-channel
    compensation → FP16 matmul using a pre-dequantized weight cache.

    This avoids BNB 4-bit dequant overhead on every forward pass (the dequant
    happens once during :func:`build_fused_weights`).  Compared to the INT8
    TensorCore approach (``torch._int_mm``), cached FP16 matmul is consistently
    faster on L20 for all layer dimensions encountered in Qwen2.5-Omni-7B.

    Architecture::

        Input [M, K]
          │
          ├─→ Per-channel asymmetric act quant+dequant
          │   (using real best_min/best_max from calibration)
          │
          ├─→ Sparse compensation (outlier channels bypass quantization)
          │
          └─→ F.linear(x_dq, W_fp16_cache)   # cached full FP16 weight

    Parameters are extracted from a *real* calibrated MorphoQuant model via
    :func:`extract_morpho_params` and :func:`build_fused_weights`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_scale: torch.Tensor,
        act_zero: torch.Tensor,
        outlier_indices: torch.Tensor,
        W_fp16_cache: torch.Tensor,
        W_outlier_fp16: Optional[torch.Tensor] = None,
        compensation_limit: Optional[torch.Tensor] = None,
        activation_bit: int = 4,
        use_cuda_kernel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bit = activation_bit
        self.qmax = float(2 ** activation_bit - 1)

        # Activation quantization parameters (per-channel, from calibration)
        self.register_buffer('act_scale', act_scale.to(torch.float32))   # [K]
        self.register_buffer('act_zero', act_zero.to(torch.float32))     # [K]

        # Outlier bypass
        self.register_buffer('outlier_indices', outlier_indices)          # [n_out] int64
        if W_outlier_fp16 is not None and W_outlier_fp16.numel() > 0:
            self.register_buffer('W_outlier_fp16', W_outlier_fp16)        # [N, n_out]
        if compensation_limit is not None:
            self.register_buffer('compensation_limit', compensation_limit.to(torch.float32))

        # ---- Pre-build outlier_mask boolean tensor for CUDA kernel (O(1) lookup) ----
        _n_out = outlier_indices.numel()
        self.has_outliers = _n_out > 0
        if self.has_outliers:
            _mask = torch.zeros(in_features, dtype=torch.bool)
            _mask[outlier_indices] = True
            self.register_buffer('_outlier_mask', _mask, persistent=True)
        else:
            self.register_buffer('_outlier_mask', torch.zeros(in_features, dtype=torch.bool),
                                 persistent=True)

        # Cached full FP16 weight (pre-dequantized from BNB 4-bit during build)
        # Stored in FP16 to save memory; converted to model dtype on first forward
        self.register_buffer('W_fp16_cache', W_fp16_cache.to(torch.float16))  # [N, K]
        self._cache_converted = False

        # ---- CUDA kernel availability ----
        self._cuda_available = False
        if use_cuda_kernel:
            try:
                from modules.morpho_cuda_kernel import is_cuda_available
                self._cuda_available = is_cuda_available()
            except Exception:
                self._cuda_available = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward: act quant → sparse compensation → FP16 matmul.

        Uses the fused CUDA kernel (:mod:`morpho_cuda_kernel`) when available,
        falling back to the pure-PyTorch element-wise path otherwise.

        Args:
            x: Input activation tensor [..., in_features] in fp16 or bf16.

        Returns:
            Output tensor [..., out_features] in the same dtype as input.
        """
        orig_shape = x.shape
        input_dtype = x.dtype
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)

        # Lazy: convert weight cache to model dtype once (avoids per-call .to())
        if not self._cache_converted:
            self.W_fp16_cache.data = self.W_fp16_cache.data.to(dtype=input_dtype)
            self._cache_converted = True

        # ---- Fast path: fused CUDA kernel ----
        if self._cuda_available:
            from modules.morpho_cuda_kernel import fused_act_quant_dequant

            comp_limit = (self.compensation_limit.to(dtype=torch.float32, device=x.device)
                          if self.has_outliers and hasattr(self, 'compensation_limit')
                          else None)
            outlier_mask = (self._outlier_mask.to(device=x.device)
                            if self.has_outliers else None)

            x_dq = fused_act_quant_dequant(
                x, self.act_scale, self.act_zero, self.qmax,
                outlier_mask, comp_limit,
            )
            y = torch.matmul(x_dq, self.W_fp16_cache.T)

            if len(orig_shape) > 2:
                y = y.reshape(*orig_shape[:-1], self.out_features)
            return y

        # ---- Fallback: pure-PyTorch path ----
        device = x.device

        # 1. Per-channel asymmetric activation quant + dequant (matching QuantAct)
        act_scale_dev = self.act_scale
        act_zero_dev = self.act_zero

        x_fp32 = x.float()
        x_scaled = x_fp32 / act_scale_dev.unsqueeze(0) + act_zero_dev.unsqueeze(0)
        x_int = torch.round(x_scaled).clamp(0, self.qmax)
        x_dq = (x_int - act_zero_dev.unsqueeze(0)) * act_scale_dev.unsqueeze(0)

        # 2. Sparse compensation: bypass outlier channels that exceed threshold
        if self.has_outliers and self.outlier_indices.numel() > 0:
            idx = self.outlier_indices
            compensation_limit = getattr(self, 'compensation_limit', None)
            if compensation_limit is not None:
                comp = compensation_limit
                above_limit = x_fp32[:, idx].abs() > comp[idx].unsqueeze(0)
                x_dq[:, idx] = torch.where(above_limit, x_fp32[:, idx], x_dq[:, idx])
            else:
                x_dq[:, idx] = x_fp32[:, idx]

        x_dq = x_dq.to(input_dtype)

        # 3. FP16 matmul with (now dtype-converted) cached full weight
        y = torch.matmul(x_dq, self.W_fp16_cache.T)

        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], self.out_features)
        return y

    def forward_fp16_baseline(self, x_fp16: torch.Tensor, W_fp16: torch.Tensor) -> torch.Tensor:
        """Reference FP16 matmul for accuracy comparison."""
        return torch.matmul(x_fp16, W_fp16.T)

    def compute_snr(self, x_fp16: torch.Tensor, W_fp16_ref: torch.Tensor) -> float:
        """Compute SNR (dB) between fused output and FP16 reference.

        Args:
            x_fp16: Input tensor.
            W_fp16_ref: Reference FP16 weight matrix [out_features, in_features].

        Returns:
            SNR in dB. Higher is better (>40 dB = negligible error).
        """
        with torch.no_grad():
            y_fused = self.forward(x_fp16)
            y_ref = self.forward_fp16_baseline(x_fp16, W_fp16_ref.to(x_fp16.device))
            signal_power = y_ref.float().pow(2).mean()
            noise_power = (y_fused.float() - y_ref.float()).pow(2).mean()
            if noise_power < 1e-12:
                return float('inf')
            snr = 10.0 * torch.log10(signal_power / noise_power)
            return snr.item()


# ---------------------------------------------------------------------------
# Recursive model replacement
# ---------------------------------------------------------------------------

def replace_morpho_with_fused(
    model: nn.Module,
    layer_params: Optional[Dict[str, dict]] = None,
    activation_bit: int = 4,
    use_cuda_kernel: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """Replace all MorphoQuant quantized layers with MorphoFusedLinear.

    Handles both ``MorphoQuantActWrapper`` and ``Linear4bit + QuantAct`` patterns.

    Args:
        model: Calibrated MorphoQuant model.
        layer_params: Pre-extracted params (if None, calls extract_morpho_params).
        activation_bit: Activation bit-width for quantization parameters.
        use_cuda_kernel: If True (default), tries the fused CUDA kernel
            (:mod:`morpho_cuda_kernel`) for the act-quant step.
        verbose: Print replacement progress.

    Returns:
        The same model with quantized layers → MorphoFusedLinear.
    """
    from bitsandbytes.nn.modules import Linear4bit
    from modules.model_factory import MorphoQuantActWrapper

    if layer_params is None:
        layer_params = extract_morpho_params(model)

    if not layer_params:
        print("[WARN] replace_morpho_with_fused: no quantized layers found.")
        return model

    replaced = 0
    skipped = 0

    for name, params in layer_params.items():
        # Build fused weights (dequant BNB 4-bit → FP16 cache once)
        try:
            act_s, act_z, W_cache, W_out = build_fused_weights(
                params, activation_bit=activation_bit
            )
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {name}: weight build failed ({e})")
            skipped += 1
            continue

        fused = MorphoFusedLinear(
            in_features=params['in_features'],
            out_features=params['out_features'],
            act_scale=act_s,
            act_zero=act_z,
            W_fp16_cache=W_cache,
            outlier_indices=params['outlier_indices'].clone(),
            W_outlier_fp16=W_out,
            compensation_limit=params.get('compensation_limit'),
            activation_bit=activation_bit,
            use_cuda_kernel=use_cuda_kernel,
        )

        # Navigate to parent module and replace
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        existing = getattr(parent, parts[-1])
        if isinstance(existing, (MorphoQuantActWrapper, Linear4bit)):
            setattr(parent, parts[-1], fused)
            replaced += 1
        else:
            skipped += 1

    if verbose:
        print(f"[MorphoFused] Replaced {replaced} layers, skipped {skipped}")

    return model
