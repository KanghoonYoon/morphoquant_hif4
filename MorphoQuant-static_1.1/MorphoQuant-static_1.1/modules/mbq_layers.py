"""
MBQ (Modality-Balanced Quantization) 伪量化 Linear 层实现（支持 W4A16 / W4A8 / W4A4）。

动机：AWQ 一类的激活感知权重量化方法在校准时把所有模态 (文本/图像/音频/视频)
的激活样本混在一起统计显著性、计算重建误差。多模态校准集里 token 数量占优的
模态（通常是文本）会主导显著通道的选择，导致视觉/音频相关权重的重建误差被
稀释，量化后多模态任务的掉点往往比纯文本任务更明显。

MBQ 的做法：校准时按模态分别累积每个输入通道的激活绝对值最大值；网格搜索
缩放系数 ratio 时，对每个模态分别计算重建误差，再取各模态误差的算术平均
（而不是按 token 数加权的池化误差），使量化决策不被 token 数最多的模态主导。

激活量化：原论文对 W4A8/W8A8 采用逐 token 对称量化（每个 token 沿特征维单独
统计 absmax），而非逐张量 (per-tensor) 单一 scale ——后者会被极少数异常 token
拉大整体 scale，把其余 token 的有效信息几乎量化到 0。本实现按论文做法采用
逐 token 对称量化。另外需注意：原论文本身只验证到 W4A8，未报告 W4A4 结果，
4-bit 激活在朴素 RTN 下崩溃是已知风险，使用 W4A4 时请预期精度可能明显下降。

参考: "MBQ: Modality-Balanced Quantization for Large Vision-Language Models" (CVPR 2025)。
"""

import torch
import torch.nn as nn

from modules.awq_layers import _fake_quant_group_asymmetric
from modules.hif4_layers import _should_skip_module

DEFAULT_MODALITIES = ("text", "multimodal")


def _fake_quant_per_token_symmetric(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """逐 token 对称伪量化：每个 token（沿最后一维以外的所有维度）单独统计 absmax。"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return x_q * scale


class MBQLinear(nn.Module):
    """MBQ 伪量化 Linear 层（权重逐组非对称量化 + 可选逐 token 对称激活量化）。

    与 AWQLinear 的区别在于校准统计与误差度量按模态分桶，finalize 时对各模态的
    重建误差取算术平均而非池化平均。act_bits=None 时为 W4A16（激活仅做缩放，
    不量化）；设置 act_bits（如 8/4）后在推理时对缩放后的激活做动态逐 token
    对称伪量化，得到 W4A8 / W4A4（注意原论文只验证到 W4A8，W4A4 精度可能明显下降）。
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        weight_bits: int = 4,
        act_bits=None,
        group_size: int = 128,
        n_grid: int = 20,
        modalities=DEFAULT_MODALITIES,
    ):
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
        self.group_size = group_size
        self.n_grid = n_grid
        self.modalities = tuple(modalities)

        self.observing = False
        self.finalized = False
        self.active_modality = self.modalities[0]

        for m in self.modalities:
            self.register_buffer(f"act_absmax_{m}", torch.zeros(self.in_features))
        self.register_buffer("scale", torch.ones(self.in_features))

    def start_observe(self):
        self.observing = True
        self.finalized = False
        for m in self.modalities:
            getattr(self, f"act_absmax_{m}").zero_()

    def set_active_modality(self, modality: str):
        self.active_modality = modality if modality in self.modalities else self.modalities[0]

    @torch.no_grad()
    def _observe(self, x: torch.Tensor):
        flat = x.detach().reshape(-1, self.in_features)
        cur_max = flat.abs().amax(dim=0)
        buf = getattr(self, f"act_absmax_{self.active_modality}")
        cur_max = cur_max.to(device=buf.device, dtype=buf.dtype)
        buf.copy_(torch.maximum(buf, cur_max))

    @torch.no_grad()
    def finalize(self, eps: float = 1e-5):
        """按模态分桶网格搜索缩放系数 ratio，使各模态重建误差的算术平均最小。"""
        stats = {
            m: getattr(self, f"act_absmax_{m}").to(self.weight.device)
            for m in self.modalities
        }
        # 只保留实际被观测到数据的模态桶，避免全零桶拉低/扭曲平均误差
        stats = {m: v for m, v in stats.items() if v.max() > 0}
        if not stats:
            stats = {self.modalities[0]: torch.ones(self.in_features, device=self.weight.device)}
        stats = {m: v.clamp(min=eps) for m, v in stats.items()}

        # 用各模态激活幅值的逐通道最大值作为缩放系数搜索的基准（与 AWQ 一致），
        # 但误差度量按模态分别计算后取平均，体现"模态均衡"。
        pooled = torch.stack(list(stats.values())).amax(dim=0)

        w = self.weight.data
        best_err, best_scale, best_w = None, None, None
        for i in range(self.n_grid + 1):
            ratio = i / self.n_grid
            s = pooled.pow(ratio).clamp(min=eps)
            s = s / s.mean()

            w_scaled = w * s.unsqueeze(0)
            w_q = _fake_quant_group_asymmetric(w_scaled, bits=self.weight_bits, group_size=self.group_size)
            w_reconstructed = w_q / s.unsqueeze(0)
            diff = w_reconstructed - w

            modal_errs = [
                (diff * act.unsqueeze(0)).pow(2).mean() for act in stats.values()
            ]
            err = torch.stack(modal_errs).mean()

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
        if self.act_bits is not None:
            x_scaled = _fake_quant_per_token_symmetric(x_scaled, bits=self.act_bits)
        return nn.functional.linear(x_scaled, self.weight, self.bias)


def replace_mbq_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    weight_bits: int = 4,
    act_bits=None,
    group_size: int = 128,
    n_grid: int = 20,
) -> int:
    """递归将 nn.Linear 替换为 MBQLinear。"""
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue

            mbq_layer = MBQLinear(
                child, weight_bits=weight_bits, act_bits=act_bits,
                group_size=group_size, n_grid=n_grid,
            )
            setattr(module, name, mbq_layer)
            count += 1
        else:
            count += replace_mbq_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings,
                weight_bits=weight_bits, act_bits=act_bits,
                group_size=group_size, n_grid=n_grid,
            )

    return count


def set_mbq_observe(model: nn.Module, enabled: bool) -> int:
    """切换模型内所有 MBQLinear 的 observe 状态（开启时重置统计量）。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, MBQLinear):
            if enabled:
                sub_module.start_observe()
            else:
                sub_module.observing = False
            count += 1
    return count


def set_mbq_modality(model: nn.Module, modality: str) -> int:
    """设置模型内所有 MBQLinear 当前校准 batch 所属的模态桶。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, MBQLinear):
            sub_module.set_active_modality(modality)
            count += 1
    return count


def finalize_mbq(model: nn.Module) -> int:
    """对所有 MBQLinear 做模态均衡的缩放系数网格搜索并完成分组权重量化。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, MBQLinear):
            sub_module.finalize()
            count += 1
    return count
