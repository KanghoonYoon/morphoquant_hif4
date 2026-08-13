"""
FreeAct: Freeing Activations for LLM Quantization (Liu et al., arXiv 2603.01776).

Structured subspace decomposition (matching paper Section 3.2 Eq. 8):
  P        = [U, U_X,   0   ]  — vision activation transform  [d, d]
  P'       = [U,   0 ,  U_X']  — text activation transform     [d, d]
  P_tilde  = [U, U_X,  U_X']^T — unified weight transform      [d, d]

where U ∈ R^{d×r} (shared), U_X ∈ R^{d×r1} (vision-unique),
U_X' ∈ R^{d×r2} (text-unique), r + r1 + r2 = d, default r1 = r2 = d/32.

Built on QuaRot Hadamard rotation (R1/R2 applied before layer replacement).
U, U_X, U_X' initialized as identity-column blocks → starts from QuaRot solution.
Optimized with AdamW (lr=1e-4, lower than paper's 1e-3 for stability) with:
  - QR re-projection every epoch (paper: "restricted to be orthogonal")
  - Identity regularization (gentle pull toward QuaRot baseline)
  - Learnable clipping thresholds (alpha parameters)
  - Per-channel scale (frozen at 1.0; infrastructure for future Kronecker)

The key ~3-line inference innovation:
  x_trans = x @ P_tilde
  Vision tokens: x_trans[:, text_unique_dims] = 0
  Text tokens:  x_trans[:, vision_unique_dims] = 0
"""

import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.hif4_layers import _should_skip_module

# Module-level registry: FreeActLinear id → root model weakref
_freeact_root_registry: dict = {}


# ---------------------------------------------------------------------------
# STE pseudo-quantization operators (with and without learnable clipping)
# ---------------------------------------------------------------------------

def _fake_quant_per_channel_symmetric(w: torch.Tensor, bits: int, dim: int = 0) -> torch.Tensor:
    """Per-channel symmetric pseudo-quantization (for weights)."""
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(w.dim()) if d != dim]
    scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / qmax
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return w + (w_q * scale - w).detach()


def _fake_quant_per_token_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token symmetric pseudo-quantization (for activations), with STE."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return x + (x_q * scale - x).detach()


def _fake_quant_weight_only(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel symmetric quantization without STE (for final weight storage)."""
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(w.dim()) if d != 0]
    scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / qmax
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return w_q * scale


# ---- Variants with learnable clipping (alpha parameter) ----

def _fake_quant_per_channel_with_clip(w: torch.Tensor, bits: int, alpha: torch.Tensor,
                                       dim: int = 0) -> torch.Tensor:
    """Per-channel symmetric pseudo-quantization with learnable clip threshold."""
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(w.dim()) if d != dim]
    raw_scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / qmax
    eff_alpha = alpha.clamp(0.1, 5.0)
    scale = raw_scale * eff_alpha
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return w + (w_q * scale - w).detach()


def _fake_quant_per_token_with_clip(x: torch.Tensor, bits: int, alpha: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric pseudo-quantization with learnable clip threshold."""
    qmax = 2 ** (bits - 1) - 1
    raw_scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    eff_alpha = alpha.clamp(0.1, 5.0)
    scale = raw_scale * eff_alpha
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return x + (x_q * scale - x).detach()


def _fake_quant_weight_only_with_clip(w: torch.Tensor, bits: int, alpha: torch.Tensor) -> torch.Tensor:
    """Per-channel symmetric quantization with clip, no STE (for final weight storage)."""
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(w.dim()) if d != 0]
    raw_scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / qmax
    eff_alpha = alpha.clamp(0.1, 5.0)
    scale = raw_scale * eff_alpha
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return w_q * scale


# ---------------------------------------------------------------------------
# Visual token count (same weakref pattern as MQuant)
# ---------------------------------------------------------------------------

def _get_freeact_visual_token_count(module: nn.Module, default: int = 0) -> int:
    """Read visual token count from model root (set by evaluator before forward)."""
    ref = _freeact_root_registry.get(id(module))
    if ref is not None:
        root = ref()
        if root is not None and hasattr(root, '_freeact_visual_token_count'):
            val = getattr(root, '_freeact_visual_token_count', None)
            if val is not None:
                return int(val)
    return default


# ---------------------------------------------------------------------------
# FreeActLinear — structured subspace decomposition
# ---------------------------------------------------------------------------

class FreeActLinear(nn.Module):
    """FreeAct pseudo-quantized Linear layer with structured subspace decomposition.

    Implements Eq. (8) from the paper:
      P        = [U, U_X,   0  ]   — vision activation transform
      P'       = [U,   0,  U_X']   — text activation transform
      P_tilde  = [U, U_X, U_X']^T  — unified weight transform (orthogonal)

    U, U_X, U_X' are learned as separate [d, r], [d, r1], [d, r2] matrices
    and assembled into P_tilde = [U, U_X, U_X'] for the weight transform.

    For layers with in_features > MAX_P_DIM, falls back to per-channel weight
    quant + per-token act quant with learnable clipping (no subspace decomposition).
    """

    MAX_P_DIM = 4096

    def __init__(
        self,
        original_linear: nn.Linear,
        weight_bits: int = 4,
        act_bits: int = 4,
        r1_ratio: float = 1.0 / 32.0,
        r2_ratio: float = 1.0 / 32.0,
        model_root: nn.Module = None,
        calib_epochs: int = 15,
        calib_lr: float = 1e-4,         # lower than paper's 1e-3 for stability
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        d = self.in_features

        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.weight_bits = weight_bits
        self.act_bits = act_bits

        # Fallback mode: for layers with d > MAX_P_DIM
        self._fallback = d > self.MAX_P_DIM

        if self._fallback:
            self.r1 = 0
            self.r2 = 0
            self.r = d
            self.U = None
            self.U_X = None
            self.U_X_prime = None
            print(f"  [FreeAct] in={d} out={self.out_features}: d > {self.MAX_P_DIM}, "
                  f"using RTN fallback (no subspace)")
        else:
            self.r1 = max(1, int(d * r1_ratio))
            self.r2 = max(1, int(d * r2_ratio))
            self.r = d - self.r1 - self.r2
            assert self.r > 0, f"Shared subspace too small: d={d}, r1={self.r1}, r2={self.r2}"

            # ---- Structured basis matrices (matching paper Eq. 8) ----
            # Initialize from identity columns: starts from QuaRot solution.
            I = torch.eye(d, dtype=original_linear.weight.dtype)
            self.U = nn.Parameter(I[:, :self.r].clone())                        # [d, r]
            self.U_X = nn.Parameter(I[:, self.r:self.r + self.r1].clone())       # [d, r1]
            self.U_X_prime = nn.Parameter(I[:, self.r + self.r1:].clone())       # [d, r2]

        # ---- Learnable clipping thresholds (init 1.0 = no extra clipping) ----
        self.act_clip_alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.weight_clip_alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # ---- Per-channel scale (FROZEN at 1.0; infrastructure for Kronecker) ----
        self.channel_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.float32),
                                          requires_grad=False)

        # Calibration hyperparams
        self.calib_epochs = calib_epochs
        self.calib_lr = calib_lr

        # State
        self.observing = False
        self.finalized = False
        self.calibrated = False

        # Collected activations during observe phase
        self._vis_acts: list = []
        self._text_acts: list = []
        self._max_calib_tokens = 4096

        # Register in weakref registry for visual token count lookup
        if model_root is not None:
            _freeact_root_registry[id(self)] = weakref.ref(model_root)

    # ---- Assemble P_tilde ------------------------------------------------

    def _assemble_P_tilde(self) -> torch.Tensor:
        """Concatenate [U, U_X, U_X_prime] into the full [d, d] weight transform."""
        return torch.cat([self.U, self.U_X, self.U_X_prime], dim=1)

    # ---- Subspace slices (for zero-masking in forward) -------------------

    @property
    def vis_unique_start(self):
        return self.r

    @property
    def vis_unique_end(self):
        return self.r + self.r1

    @property
    def text_unique_start(self):
        return self.r + self.r1

    @property
    def text_unique_end(self):
        return self.r + self.r1 + self.r2

    # ---- Observe ---------------------------------------------------------

    def start_observe(self):
        """Begin collecting activations for calibration."""
        self.observing = True
        self.finalized = False
        self.calibrated = False
        self._vis_acts.clear()
        self._text_acts.clear()

    @torch.no_grad()
    def _observe(self, x: torch.Tensor):
        """Store input activations, split by token type."""
        vis_total = sum(a.size(0) for a in self._vis_acts)
        text_total = sum(a.size(0) for a in self._text_acts)
        if vis_total >= self._max_calib_tokens and text_total >= self._max_calib_tokens:
            return

        if x.dim() == 3:
            b, s, d = x.shape
            v_cnt = _get_freeact_visual_token_count(self, default=0)
            v_cnt = min(v_cnt, s)

            if v_cnt > 0 and vis_total < self._max_calib_tokens:
                vis_tokens = x[:, :v_cnt, :].reshape(-1, d)
                remaining = self._max_calib_tokens - vis_total
                if vis_tokens.size(0) > remaining:
                    vis_tokens = vis_tokens[:remaining]
                self._vis_acts.append(vis_tokens.detach().cpu())

            if v_cnt < s and text_total < self._max_calib_tokens:
                text_tokens = x[:, v_cnt:, :].reshape(-1, d)
                remaining = self._max_calib_tokens - text_total
                if text_tokens.size(0) > remaining:
                    text_tokens = text_tokens[:remaining]
                self._text_acts.append(text_tokens.detach().cpu())

            if v_cnt == 0 or v_cnt >= s:
                flat = x.reshape(-1, d)
                remaining = self._max_calib_tokens - text_total
                if flat.size(0) > remaining:
                    flat = flat[:remaining]
                if text_total < self._max_calib_tokens:
                    self._text_acts.append(flat.detach().cpu())
        else:
            flat = x.reshape(-1, d)
            remaining = self._max_calib_tokens - text_total
            if flat.size(0) > remaining:
                flat = flat[:remaining]
            if text_total < self._max_calib_tokens:
                self._text_acts.append(flat.detach().cpu())

    # ---- Calibration (layer-wise AdamW optimization) --------------------

    def calibrate_layer(self, device: torch.device, dtype: torch.dtype):
        """Optimize U, U_X, U_X_prime, and clip thresholds with AdamW.

        Minimizes MSE between FP output and quantized output:
          L = MSE(X_vis@W, Q(X_vis @ [U, U_X, 0])   @ Q(W @ [U,U_X,U_X']^T))
            + MSE(X_text@W, Q(X_text @ [U, 0, U_X']) @ Q(W @ [U,U_X,U_X']^T))

        with orthogonality + identity regularization on [U, U_X, U_X'].
        QR re-projection applied EVERY epoch for strict orthogonality.
        Lower lr (1e-4 vs paper's 1e-3) prevents overfitting to calibration data.
        """
        if not self._vis_acts and not self._text_acts:
            print(f"  [FreeAct] WARNING: No calibration data for "
                  f"in={self.in_features} out={self.out_features}. Skipping.")
            return

        vis_acts = torch.cat(self._vis_acts, dim=0).to(device=device, dtype=dtype) if self._vis_acts else None
        text_acts = torch.cat(self._text_acts, dim=0).to(device=device, dtype=dtype) if self._text_acts else None
        if vis_acts is None:
            vis_acts = text_acts
        if text_acts is None:
            text_acts = vis_acts

        self._to_device(device, dtype)

        # Teacher outputs (FP reference)
        with torch.no_grad():
            teacher_vis = F.linear(vis_acts, self.weight, self.bias)
            teacher_text = F.linear(text_acts, self.weight, self.bias)

        # Build optimizer params: U, U_X, U_X_prime, clip alphas
        optim_params = [self.act_clip_alpha, self.weight_clip_alpha]
        if not self._fallback:
            optim_params.extend([self.U, self.U_X, self.U_X_prime])

        optimizer = torch.optim.AdamW(optim_params, lr=self.calib_lr)

        d = self.in_features
        if not self._fallback:
            r, r1, r2 = self.r, self.r1, self.r2

        for epoch in range(self.calib_epochs):
            optimizer.zero_grad()

            cs = self.channel_scale.to(dtype=dtype)  # frozen at 1.0
            vis_scaled = vis_acts * cs
            text_scaled = text_acts * cs
            W_scaled = self.weight / cs

            # ---- Weight transform & quantize ----
            if self._fallback:
                W_q = _fake_quant_per_channel_with_clip(
                    W_scaled, bits=self.weight_bits, alpha=self.weight_clip_alpha, dim=0)
            else:
                P_tilde = self._assemble_P_tilde()   # [d, d]
                W_trans = W_scaled @ P_tilde          # [out, d]
                W_q = _fake_quant_per_channel_with_clip(
                    W_trans, bits=self.weight_bits, alpha=self.weight_clip_alpha, dim=0)

            loss = 0.0

            if self._fallback:
                x_vis_q = _fake_quant_per_token_with_clip(
                    vis_scaled, bits=self.act_bits, alpha=self.act_clip_alpha)
                out_vis = F.linear(x_vis_q, W_q, self.bias)
                loss = loss + F.mse_loss(out_vis, teacher_vis)

                x_text_q = _fake_quant_per_token_with_clip(
                    text_scaled, bits=self.act_bits, alpha=self.act_clip_alpha)
                out_text = F.linear(x_text_q, W_q, self.bias)
                loss = loss + F.mse_loss(out_text, teacher_text)
            else:
                # ---- Vision: x @ [U, U_X, 0] (zero text-unique dims) ----
                x_vis_trans = vis_scaled @ P_tilde          # [N_vis, d]
                x_vis_trans[:, r + r1:] = 0                  # zero text-unique (last r2 dims)
                x_vis_q = _fake_quant_per_token_with_clip(
                    x_vis_trans, bits=self.act_bits, alpha=self.act_clip_alpha)
                out_vis = F.linear(x_vis_q, W_q, self.bias)
                loss = loss + F.mse_loss(out_vis, teacher_vis)

                # ---- Text: x @ [U, 0, U_X'] (zero vision-unique dims) ----
                x_text_trans = text_scaled @ P_tilde        # [N_text, d]
                x_text_trans[:, r:r + r1] = 0               # zero vision-unique (middle r1 dims)
                x_text_q = _fake_quant_per_token_with_clip(
                    x_text_trans, bits=self.act_bits, alpha=self.act_clip_alpha)
                out_text = F.linear(x_text_q, W_q, self.bias)
                loss = loss + F.mse_loss(out_text, teacher_text)

                # ---- Regularization ----
                Pt = P_tilde.to(torch.float32)
                I_d = torch.eye(d, device=device, dtype=torch.float32)

                # Orthogonality: ||P_tilde^T @ P_tilde - I||
                ortho_reg = F.mse_loss(Pt.T @ Pt, I_d)
                loss = loss + 0.1 * ortho_reg

                # Identity pull: ||P_tilde - I|| (gentle, keeps near QuaRot baseline)
                ident_reg = F.mse_loss(Pt, I_d)
                loss = loss + 0.01 * ident_reg

            loss.backward()
            optimizer.step()

            # ---- Hard QR re-projection EVERY epoch ----
            if not self._fallback:
                self._qr_project()

            if (epoch + 1) % 5 == 0:
                print(f"    [FreeAct] in={self.in_features} out={self.out_features} "
                      f"epoch {epoch+1}/{self.calib_epochs} loss={loss.item():.6f}")

        # ---- Post-calibration cleanup ----
        del optimizer, teacher_vis, teacher_text, vis_acts, text_acts
        self._vis_acts.clear()
        self._text_acts.clear()
        self._to_cpu()
        self.calibrated = True
        torch.cuda.empty_cache()

        print(f"  [FreeAct] Calibrated in={self.in_features} out={self.out_features} "
              f"(r={self.r} r1={self.r1} r2={self.r2})")

    # ---- QR re-projection ------------------------------------------------

    @torch.no_grad()
    def _qr_project(self):
        """QR-project [U, U_X, U_X_prime] to be strictly orthogonal."""
        if self._fallback:
            return
        dtype = self.U.dtype
        # Concatenate into [d, d]
        P = torch.cat([self.U.data, self.U_X.data, self.U_X_prime.data], dim=1)
        Pf = P.to(torch.float32)
        Q, R = torch.linalg.qr(Pf)
        # Ensure positive diagonal of R
        sign = torch.diag(torch.sign(torch.diag(R)))
        Q = Q @ sign
        # Split back into sub-blocks
        self.U.data = Q[:, :self.r].to(dtype=dtype)
        self.U_X.data = Q[:, self.r:self.r + self.r1].to(dtype=dtype)
        self.U_X_prime.data = Q[:, self.r + self.r1:].to(dtype=dtype)

    # ---- Device helpers --------------------------------------------------

    def _to_device(self, device: torch.device, dtype: torch.dtype):
        """Move all learnable params and weight to the given device."""
        self.weight.data = self.weight.data.to(device=device, dtype=dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(device=device, dtype=dtype)
        for p in [self.U, self.U_X, self.U_X_prime]:
            if p is not None:
                p.data = p.data.to(device=device, dtype=dtype)
        self.channel_scale.data = self.channel_scale.data.to(device=device)
        self.act_clip_alpha.data = self.act_clip_alpha.data.to(device=device)
        self.weight_clip_alpha.data = self.weight_clip_alpha.data.to(device=device)

    def _to_cpu(self):
        """Move subspace params and clip alphas to CPU. Weight stays on GPU."""
        for p in [self.U, self.U_X, self.U_X_prime]:
            if p is not None:
                p.data = p.data.cpu()
        self.channel_scale.data = self.channel_scale.data.cpu()
        self.act_clip_alpha.data = self.act_clip_alpha.data.cpu()
        self.weight_clip_alpha.data = self.weight_clip_alpha.data.cpu()

    def _lazy_to_device(self, x: torch.Tensor):
        """Move params to x's device if needed."""
        for p in [self.U, self.U_X, self.U_X_prime,
                  self.channel_scale, self.act_clip_alpha, self.weight_clip_alpha]:
            if p is not None and p.device != x.device:
                p.data = p.data.to(device=x.device)

    # ---- Finalize --------------------------------------------------------

    @torch.no_grad()
    def finalize(self):
        """Bake channel_scale and P_tilde into quantized weight.

        W_final = quantize_with_clip((W / channel_scale) @ P_tilde)
        For fallback: W_final = quantize_with_clip(W / channel_scale)
        """
        if self.finalized:
            return

        device = self.weight.data.device
        self._lazy_to_device(self.weight.data)

        dtype = self.weight.data.dtype
        cs = self.channel_scale.to(dtype=dtype)
        W_scaled = self.weight.data / cs

        if self._fallback:
            self.weight.data = _fake_quant_weight_only_with_clip(
                W_scaled, bits=self.weight_bits, alpha=self.weight_clip_alpha)
        else:
            P_tilde = self._assemble_P_tilde()
            W_trans = W_scaled @ P_tilde
            self.weight.data = _fake_quant_weight_only_with_clip(
                W_trans, bits=self.weight_bits, alpha=self.weight_clip_alpha)

        self._to_cpu()
        self.observing = False
        self.finalized = True

    # ---- Forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self._observe(x)
            return F.linear(x, self.weight, self.bias)

        if not self.finalized:
            return F.linear(x, self.weight, self.bias)

        self._lazy_to_device(x)
        cs = self.channel_scale.to(dtype=x.dtype)

        if self._fallback:
            x_scaled = x * cs
            x_q = _fake_quant_per_token_with_clip(
                x_scaled, bits=self.act_bits, alpha=self.act_clip_alpha)
            return F.linear(x_q, self.weight, self.bias)

        # --- FreeAct structured inference ---

        # 1. Per-channel scale (cs ≈ 1.0, frozen)
        x_scaled = x * cs

        # 2. Full P_tilde transform: x @ [U, U_X, U_X_prime]
        P_tilde = self._assemble_P_tilde()
        x_trans = x_scaled @ P_tilde

        # 3. Zero-mask unique-subspace output dims based on token type
        if x_trans.dim() == 3:
            b, s, d = x_trans.shape
            v_cnt = _get_freeact_visual_token_count(self, default=0)
            v_cnt = min(v_cnt, s)

            if v_cnt > 0:
                # Vision tokens: zero out text-unique dims (last r2 cols)
                x_trans[:, :v_cnt, self.text_unique_start:self.text_unique_end] = 0

            if v_cnt < s:
                # Text tokens: zero out vision-unique dims (middle r1 cols)
                x_trans[:, v_cnt:, self.vis_unique_start:self.vis_unique_end] = 0

        # 4. Per-token quantize activations (with learned clipping)
        x_q = _fake_quant_per_token_with_clip(
            x_trans, bits=self.act_bits, alpha=self.act_clip_alpha)

        # 5. Matmul with final quantized weight
        return F.linear(x_q, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Recursive replacement
# ---------------------------------------------------------------------------

def replace_freeact_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    weight_bits: int = 4,
    act_bits: int = 4,
    r1_ratio: float = 1.0 / 32.0,
    r2_ratio: float = 1.0 / 32.0,
    model_root: nn.Module = None,
    calib_epochs: int = 15,
    calib_lr: float = 1e-4,
) -> int:
    """Recursively replace nn.Linear with FreeActLinear."""
    if model_root is None:
        model_root = module

    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue
            fa_layer = FreeActLinear(
                child,
                weight_bits=weight_bits,
                act_bits=act_bits,
                r1_ratio=r1_ratio,
                r2_ratio=r2_ratio,
                model_root=model_root,
                calib_epochs=calib_epochs,
                calib_lr=calib_lr,
            )
            setattr(module, name, fa_layer)
            count += 1
        else:
            count += replace_freeact_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings,
                weight_bits=weight_bits, act_bits=act_bits,
                r1_ratio=r1_ratio, r2_ratio=r2_ratio,
                model_root=model_root,
                calib_epochs=calib_epochs, calib_lr=calib_lr,
            )
    return count


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------

def set_freeact_observe(model: nn.Module, enabled: bool):
    """Toggle observe mode on all FreeActLinear layers."""
    for module in model.modules():
        if isinstance(module, FreeActLinear):
            if enabled:
                module.start_observe()
            else:
                module.observing = False


def finalize_freeact(model: nn.Module):
    """Run per-layer AdamW calibration + weight quantization for all FreeActLinear layers."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, FreeActLinear)]

    print(f"\n[FreeAct] Starting per-layer AdamW calibration for {len(layers)} layers...")
    for i, (name, layer) in enumerate(layers):
        print(f"  [{i+1}/{len(layers)}] {name}")
        layer.calibrate_layer(device=device, dtype=dtype)
        layer.finalize()

    print(f"[FreeAct] Calibration complete — {len(layers)} layers calibrated and quantized.\n")
