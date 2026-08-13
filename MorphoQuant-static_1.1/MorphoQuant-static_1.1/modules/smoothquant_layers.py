"""
SmoothQuant (W8A8) 伪量化层实现。

流程:
  1. Observe  — 用校准数据跑前向，统计每个输入通道激活值的绝对值最大值
  2. Finalize — 结合权重的逐输入通道绝对值最大值，计算平滑系数 s
                (s_j = max|X_j|^alpha / max|W_j|^(1-alpha))，
                把 s 融合进权重 (w' = w * s)，并据此对平滑后的权重做
                一次性逐输出通道 int8 伪量化（静态）
  3. Quant    — 推理时，对激活先除以 s 做平滑，再做逐张量 int8 伪量化
                （动态），最后与已伪量化的权重做线性变换

参考: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models" (ICML 2023)。
"""

import torch
import torch.nn as nn

from modules.hif4_layers import _should_skip_module


def _fake_quant_per_tensor_symmetric(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return x_q * scale


def _fake_quant_per_channel_symmetric(w: torch.Tensor, bits: int = 8, dim: int = 0) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(w.dim()) if d != dim]
    scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / qmax
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return w_q * scale


class SmoothQuantLinear(nn.Module):
    """SmoothQuant 伪量化 Linear 层（纯 SmoothQuant 路径，不耦合 MorphoQuant）。

    通过 weight_bits / act_bits 支持 W8A8 / W4A8 / W4A16 三种位宽组合：
      - act_bits=None 时不对激活做伪量化（WxA16），但仍使用观测到的激活幅值
        计算平滑系数，使权重量化更友好。
    """

    def __init__(self, original_linear: nn.Linear, weight_bits: int = 8, act_bits=8):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.weight_bits = weight_bits
        self.act_bits = act_bits

        self.observing = False
        self.finalized = False
        self.register_buffer("act_absmax", torch.zeros(self.in_features))
        self.register_buffer("smooth_scale", torch.ones(self.in_features))

    def start_observe(self):
        self.observing = True
        self.finalized = False
        self.act_absmax.zero_()

    @torch.no_grad()
    def _observe(self, x: torch.Tensor):
        flat = x.detach().reshape(-1, self.in_features)
        cur_max = flat.abs().amax(dim=0).to(device=self.act_absmax.device, dtype=self.act_absmax.dtype)
        self.act_absmax = torch.maximum(self.act_absmax, cur_max)

    @torch.no_grad()
    def finalize(self, alpha: float = 0.5, eps: float = 1e-5):
        """结合观测到的激活幅值与权重幅值计算平滑系数，并对权重做一次性逐通道量化。"""
        weight_absmax = self.weight.detach().abs().amax(dim=0).clamp(min=eps)  # [in_features]
        act_absmax = self.act_absmax.to(self.weight.device).clamp(min=eps)

        scale = (act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)).clamp(min=eps)
        self.smooth_scale = scale.to(dtype=self.weight.dtype, device=self.weight.device)

        smoothed_weight = self.weight.data * self.smooth_scale.unsqueeze(0)
        self.weight.data = _fake_quant_per_channel_symmetric(smoothed_weight, bits=self.weight_bits, dim=0)

        self.observing = False
        self.finalized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self._observe(x)
            return nn.functional.linear(x, self.weight, self.bias)

        if not self.finalized:
            return nn.functional.linear(x, self.weight, self.bias)

        x_smooth = x / self.smooth_scale.to(x.dtype)
        if self.act_bits is not None:
            x_smooth = _fake_quant_per_tensor_symmetric(x_smooth, bits=self.act_bits)
        return nn.functional.linear(x_smooth, self.weight, self.bias)


def replace_smoothquant_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    weight_bits: int = 8,
    act_bits: int = 8,
) -> int:
    """递归将 nn.Linear 替换为 SmoothQuantLinear（纯 SmoothQuant 伪量化路径）。"""
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue

            sq_layer = SmoothQuantLinear(child, weight_bits=weight_bits, act_bits=act_bits)
            setattr(module, name, sq_layer)
            count += 1
        else:
            count += replace_smoothquant_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings,
                weight_bits=weight_bits, act_bits=act_bits,
            )

    return count


def set_smoothquant_observe(model: nn.Module, enabled: bool) -> int:
    """切换模型内所有 SmoothQuantLinear 的 observe 状态（开启时重置统计量）。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, SmoothQuantLinear):
            if enabled:
                sub_module.start_observe()
            else:
                sub_module.observing = False
            count += 1
    return count


def finalize_smoothquant(model: nn.Module, alpha: float = 0.5) -> int:
    """对所有 SmoothQuantLinear 计算平滑系数并完成权重量化。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, SmoothQuantLinear):
            sub_module.finalize(alpha=alpha)
            count += 1
    return count
