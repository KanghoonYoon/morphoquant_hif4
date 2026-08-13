"""
AWQ (Activation-aware Weight Quantization) 伪量化 Linear 层实现（W4A16）。

流程:
  1. Observe  — 用校准数据跑前向，统计每个输入通道激活值的绝对值最大值，
                作为该输入通道"显著性" (saliency) 的代理指标。
  2. Finalize — 在若干候选 ratio 上做网格搜索：对每个 ratio 计算逐输入通道
                缩放系数 s_j = max|X_j|^ratio，将其乘入权重 (w' = w * s)，
                再做逐输出通道、组内 (group_size) 非对称 INT 伪量化；
                选择使重建误差（按激活幅值加权）最小的 ratio，并把对应的
                缩放系数与组量化权重保存下来。
  3. Quant    — 推理时，对激活先除以 s 做缩放，再与已伪量化的权重做线性变换
                (W4A16: 激活本身保持原精度，不做量化)。

参考: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration" (MLSys 2024)。
"""

import torch
import torch.nn as nn

from modules.hif4_layers import _should_skip_module


def _fake_quant_group_asymmetric(w: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """逐输出通道、组内 (沿输入维度切分) 非对称量化-反量化。"""
    out_features, in_features = w.shape
    qmax = 2 ** bits - 1

    gs = group_size if 0 < group_size < in_features else in_features
    pad = (-in_features) % gs
    w_pad = nn.functional.pad(w, (0, pad)) if pad else w

    w_g = w_pad.view(out_features, -1, gs)
    w_min = w_g.amin(dim=-1, keepdim=True)
    w_max = w_g.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min).clamp(min=1e-8) / qmax
    zero = (-w_min / scale).round()

    w_q = torch.clamp(torch.round(w_g / scale) + zero, 0, qmax)
    w_dq = (w_q - zero) * scale
    w_dq = w_dq.reshape(out_features, -1)
    if pad:
        w_dq = w_dq[:, :in_features]
    return w_dq


class AWQLinear(nn.Module):
    """AWQ 伪量化 Linear 层（纯 AWQ 路径，不耦合 MorphoQuant）。

    仅支持权重量化 (WxA16)：激活始终保持原精度，与 AWQ 论文中
    "保护显著激活通道对应的权重精度" 的核心思想一致。
    """

    def __init__(self, original_linear: nn.Linear, weight_bits: int = 4, group_size: int = 128, n_grid: int = 20):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.n_grid = n_grid

        self.observing = False
        self.finalized = False
        self.register_buffer("act_absmax", torch.zeros(self.in_features))
        self.register_buffer("scale", torch.ones(self.in_features))

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
    def finalize(self, eps: float = 1e-5):
        """网格搜索激活感知缩放系数 ratio，并据此完成分组非对称权重量化。"""
        act_absmax = self.act_absmax.to(self.weight.device).clamp(min=eps)
        w = self.weight.data

        best_err, best_scale, best_w = None, None, None
        for i in range(self.n_grid + 1):
            ratio = i / self.n_grid
            s = act_absmax.pow(ratio).clamp(min=eps)
            s = s / s.mean()

            w_scaled = w * s.unsqueeze(0)
            w_q = _fake_quant_group_asymmetric(w_scaled, bits=self.weight_bits, group_size=self.group_size)
            w_reconstructed = w_q / s.unsqueeze(0)

            err = ((w_reconstructed - w) * act_absmax.unsqueeze(0)).pow(2).mean()
            if best_err is None or err < best_err:
                best_err, best_scale, best_w = err, s, w_q

        self.scale = best_scale.to(dtype=self.weight.dtype, device=self.weight.device)
        self.weight.data = best_w.to(dtype=self.weight.dtype)

        self.observing = False
        self.finalized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self._observe(x)
            return nn.functional.linear(x, self.weight, self.bias)

        if not self.finalized:
            return nn.functional.linear(x, self.weight, self.bias)

        x_scaled = x / self.scale.to(x.dtype)
        return nn.functional.linear(x_scaled, self.weight, self.bias)


def replace_awq_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    weight_bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
) -> int:
    """递归将 nn.Linear 替换为 AWQLinear（纯 AWQ 伪量化路径）。"""
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue

            awq_layer = AWQLinear(child, weight_bits=weight_bits, group_size=group_size, n_grid=n_grid)
            setattr(module, name, awq_layer)
            count += 1
        else:
            count += replace_awq_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings,
                weight_bits=weight_bits, group_size=group_size, n_grid=n_grid,
            )

    return count


def set_awq_observe(model: nn.Module, enabled: bool) -> int:
    """切换模型内所有 AWQLinear 的 observe 状态（开启时重置统计量）。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, AWQLinear):
            if enabled:
                sub_module.start_observe()
            else:
                sub_module.observing = False
            count += 1
    return count


def finalize_awq(model: nn.Module) -> int:
    """对所有 AWQLinear 做缩放系数网格搜索并完成分组权重量化。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, AWQLinear):
            sub_module.finalize()
            count += 1
    return count
