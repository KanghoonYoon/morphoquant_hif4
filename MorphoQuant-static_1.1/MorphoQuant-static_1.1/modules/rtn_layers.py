"""
RTN (Round-To-Nearest) 伪量化 Linear 层实现。

提供两种无需校准数据的纯权重 / 权重+激活伪量化路径：
  - W4A16: 仅对权重做逐输出通道 4bit（可配置）RTN 伪量化，激活保持原精度
  - W4A8 : 在 W4A16 基础上，额外对激活做逐张量动态 8bit（可配置）RTN 伪量化

与 modules/hif4_layers.py 中 HiF4Linear 思路一致：首次 forward 时惰性完成
一次性权重伪量化，之后激活（如启用）在每次 forward 时动态伪量化。
"""

import torch
import torch.nn as nn

from modules.hif4_layers import _should_skip_module


def _fake_quant_per_channel_symmetric(w: torch.Tensor, bits: int, dim: int = 0) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(w.dim()) if d != dim]
    scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / qmax
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return w_q * scale


def _fake_quant_per_tensor_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return x_q * scale


class RTNQuantLinear(nn.Module):
    """RTN 伪量化 Linear 层：权重逐输出通道伪量化 + 可选激活逐张量动态伪量化。"""

    def __init__(self, original_linear: nn.Linear, weight_bits: int = 4, act_bits=None):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.weight_bits = weight_bits
        self.act_bits = act_bits  # None 表示不量化激活 (WxA16)
        self.weight_quantized = False

    def quantize_weight(self):
        if self.weight_quantized:
            return
        with torch.no_grad():
            self.weight.data = _fake_quant_per_channel_symmetric(self.weight.data, bits=self.weight_bits, dim=0)
        self.weight_quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.weight_quantized:
            self.quantize_weight()

        if self.act_bits is not None:
            x = _fake_quant_per_tensor_symmetric(x, bits=self.act_bits)

        return nn.functional.linear(x, self.weight, self.bias)


def replace_rtn_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    weight_bits: int = 4,
    act_bits=None,
) -> int:
    """递归将 nn.Linear 替换为 RTNQuantLinear。"""
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue

            rtn_layer = RTNQuantLinear(child, weight_bits=weight_bits, act_bits=act_bits)
            setattr(module, name, rtn_layer)
            count += 1
        else:
            count += replace_rtn_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings,
                weight_bits=weight_bits, act_bits=act_bits,
            )

    return count
