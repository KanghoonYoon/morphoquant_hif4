"""
MQuant 伪量化 Linear 层实现（W4A4 全静态量化 + 模态特化 MSQ）。

参考: "MQuant: Unleashing the Inference Potential of Multimodal Large
Language Models via Full Static Quantization" (ACM MM 2025).

核心机制
--------
1. MSQ (Modality-Specific Static Quantization)
   视觉 token 与文本 token 的激活分布差异显著（视觉 token 分布在 ~[-20,10]，
   文本 token 集中在 0 附近）。校准阶段按模态分别统计逐通道 absmax，推理时
   对两类 token 使用各自独立的量化 scale，避免文本 token 的微弱信号被视觉
   token 的大幅值"淹没"。

2. 权重量化
   逐输出通道对称量化（per-channel symmetric），在校准完成后一次性完成。

3. 激活量化
   逐 token 对称量化（per-token symmetric），每个 token 沿特征维单独统计
   absmax。推理时按 token 所属模态（视觉/文本）选择对应的静态 scale。

用法（与 SmoothQuant / AWQ / MBQ 一致）
-----------------------------------------
1. 模型加载后调用 replace_mquant_layers_recursive() 替换 Linear 层
2. calibrate 阶段: set_mquant_observe(model, True) → 跑校准数据
3. prepare 阶段: finalize_mquant(model) → 计算 scale 并量化权重
4. evaluate 阶段: 推理前设置 model._mquant_visual_token_count 告知视觉 token 数量
"""

import weakref

import torch
import torch.nn as nn

from modules.hif4_layers import _should_skip_module

# 模块级注册表：MQuantLinear id → root model weakref
# 避免将 root nn.Module 直接存储为子模块属性（会导致 _apply() 递归溢出）
_mquant_root_registry: dict = {}


# ---------------------------------------------------------------------------
# 基础伪量化算子
# ---------------------------------------------------------------------------

def _fake_quant_per_channel_symmetric(w: torch.Tensor, bits: int, dim: int = 0) -> torch.Tensor:
    """逐通道对称伪量化（用于权重量化）。"""
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(w.dim()) if d != dim]
    scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / qmax
    w_q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return w_q * scale


def _fake_quant_per_token_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """逐 token 对称伪量化（用于激活量化）。"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return x_q * scale


# ---------------------------------------------------------------------------
# MQuantLinear
# ---------------------------------------------------------------------------

class MQuantLinear(nn.Module):
    """MQuant 全静态伪量化 Linear 层（W4A4 + 模态特化 MSQ + 可选 RMS）。

    act_bits=None 时不做激活量化（仅对权重做 W4 伪量化，等价于 W4A16 的朴素 RTN）；
    设 act_bits=4 即 W4A4 全静态量化（带模态特化 scale）。

    RMS (Rotation Magnitude Suppression): 当 use_rms=True 时，finalize() 会自动
    检测第一输入通道是否因 Hadamard 旋转产生异常大量级；若是，则 forward 时将
    第一通道分离为 full-precision bf16 路径，其余通道正常 W4A4 量化。
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        weight_bits: int = 4,
        act_bits: int = 4,
        model_root: nn.Module = None,
        use_rms: bool = False,
        rms_threshold: float = 3.0,
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

        # 通过模块级注册表保存 root model 引用（weakref），避免 nn.Module 循环引用
        if model_root is not None:
            _mquant_root_registry[id(self)] = weakref.ref(model_root)

        # RMS 配置
        self._use_rms_candidate = use_rms   # 候选标记（config 传入）
        self._rms_threshold = rms_threshold  # auto-detect 阈值
        self._rms_enabled = False            # finalize 时 auto-detect 决定

        # 状态机
        self.observing = False
        self.finalized = False

        # 校准期逐通道 absmax 统计 —— 视觉 / 文本 分别记录
        self.register_buffer("vis_absmax", torch.zeros(self.in_features))
        self.register_buffer("text_absmax", torch.zeros(self.in_features))

        # 模态特化 scale —— finalize 时计算
        self.register_buffer("vis_scale", torch.ones(1))
        self.register_buffer("text_scale", torch.ones(1))

        # 向后兼容的统一 scale（用于无法区分模态的场景）
        self.register_buffer("scale", torch.ones(1))

    # ---- observe -----------------------------------------------------------

    def start_observe(self):
        self.observing = True
        self.finalized = False
        self.vis_absmax.zero_()
        self.text_absmax.zero_()

    @torch.no_grad()
    def _observe(self, x: torch.Tensor):
        """按 token 模态分桶累积逐通道 absmax。"""
        # x: [batch, seq_len, hidden_dim] 或 [batch*tokens, hidden_dim]
        if x.dim() == 3:
            b, s, d = x.shape
            v_cnt = _get_visual_token_count(self, default=0)
            v_cnt = min(v_cnt, s)

            if v_cnt > 0:
                vis_tokens = x[:, :v_cnt, :].reshape(-1, d)
                cur_vis = vis_tokens.abs().amax(dim=0)
                cur_vis = cur_vis.to(device=self.vis_absmax.device, dtype=self.vis_absmax.dtype)
                self.vis_absmax = torch.maximum(self.vis_absmax, cur_vis)

            if v_cnt < s:
                text_tokens = x[:, v_cnt:, :].reshape(-1, d)
                cur_text = text_tokens.abs().amax(dim=0)
                cur_text = cur_text.to(device=self.text_absmax.device, dtype=self.text_absmax.dtype)
                self.text_absmax = torch.maximum(self.text_absmax, cur_text)

            if v_cnt == 0 or v_cnt >= s:
                # 无法区分（仅单一模态），退化为统一统计
                flat = x.reshape(-1, d)
                cur = flat.abs().amax(dim=0).to(device=self.text_absmax.device, dtype=self.text_absmax.dtype)
                self.text_absmax = torch.maximum(self.text_absmax, cur)
                self.vis_absmax = torch.maximum(self.vis_absmax, cur)
        else:
            # 2D fallback
            flat = x.reshape(-1, self.in_features)
            cur = flat.abs().amax(dim=0).to(device=self.text_absmax.device, dtype=self.text_absmax.dtype)
            self.text_absmax = torch.maximum(self.text_absmax, cur)
            self.vis_absmax = torch.maximum(self.vis_absmax, cur)

    # ---- finalize ----------------------------------------------------------

    @torch.no_grad()
    def finalize(self, eps: float = 1e-5):
        """基于校准期统计计算模态特化 scale 并量化权重。

        激活 scale 的计算方式：
          scale_modality = absmax_modality / (2^(act_bits-1) - 1)
        其中 absmax_modality 是该校准期内该模态所有 token 逐通道 absmax 的
        最大值（即取通道维度上的 amax），得到一个标量。

        权重做逐输出通道对称量化。

        RMS auto-detect: 若第一通道的 absmax 显著大于其余通道中位数，
        则该层启用 RMS（第一通道 full precision，其余正常 W4A4）。
        """
        qmax_act = (2 ** (self.act_bits - 1) - 1) if self.act_bits is not None else 1.0

        # ---- RMS auto-detect (基于权重统计，而非激活统计) ----
        # Hadamard R1 旋转后 consumer 层有确定性数学性质:
        #   W_new[:, 0] = √n * mean(W_old, dim=1)
        # 第一列的 L2 norm 通常远大于其余列中位数。
        # producer 层 (R1^T @ W) 和无旋转层不受影响，不会误触发。
        if self._use_rms_candidate and self.act_bits is not None and self.in_features > 1:
            w = self.weight.data.float()
            col_norms = w.norm(dim=0)  # (in_features,) L2 norm per input channel
            ch0_norm = col_norms[0].item()
            rest_median_norm = col_norms[1:].median().item()

            if ch0_norm > self._rms_threshold * max(rest_median_norm, eps):
                self._rms_enabled = True
                if not hasattr(MQuantLinear, '_rms_debug_count'):
                    MQuantLinear._rms_debug_count = 0
                if MQuantLinear._rms_debug_count < 10:
                    MQuantLinear._rms_debug_count += 1
                    print(f"  [RMS] layer triggered (weight-based): in={self.in_features} out={self.out_features} "
                          f"ch0_norm={ch0_norm:.2f} rest_median={rest_median_norm:.2f} "
                          f"ratio={ch0_norm/max(rest_median_norm,eps):.1f}x")

        # ---- MSQ scale 计算 ----
        if self._rms_enabled:
            # 用 ch1: (排除第一通道) 计算 scale，避免第一通道的巨大量级主导 scale
            vis_amax = self.vis_absmax[1:].to(self.weight.device).max().clamp(min=eps)
            text_amax = self.text_absmax[1:].to(self.weight.device).max().clamp(min=eps)
        else:
            vis_amax = self.vis_absmax.to(self.weight.device).max().clamp(min=eps)
            text_amax = self.text_absmax.to(self.weight.device).max().clamp(min=eps)

        self.vis_scale = (vis_amax / qmax_act).to(dtype=self.weight.dtype)
        self.text_scale = (text_amax / qmax_act).to(dtype=self.weight.dtype)

        # 统一 scale（fallback）
        unified_amax = torch.max(vis_amax, text_amax)
        self.scale = (unified_amax / qmax_act).to(dtype=self.weight.dtype)

        # ---- 权重量化 ----
        if self._rms_enabled:
            # W[:, 0] 保持 full precision, W[:, 1:] 逐通道 W4 量化
            w_rest = self.weight.data[:, 1:]
            w_rest_q = _fake_quant_per_channel_symmetric(w_rest, bits=self.weight_bits, dim=0)
            self.weight.data[:, 1:] = w_rest_q
        else:
            self.weight.data = _fake_quant_per_channel_symmetric(
                self.weight.data, bits=self.weight_bits, dim=0
            )

        self.observing = False
        self.finalized = True

    # ---- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self._observe(x)
            return nn.functional.linear(x, self.weight, self.bias)

        if not self.finalized:
            return nn.functional.linear(x, self.weight, self.bias)

        if self.act_bits is None:
            return nn.functional.linear(x, self.weight, self.bias)

        # ---- RMS 推理路径 ----
        if self._rms_enabled:
            return self._forward_rms(x)

        # ---- MSQ 推理路径 ----
        if x.dim() == 3:
            b, s, d = x.shape
            v_cnt = _get_visual_token_count(self, default=0)
            v_cnt = min(v_cnt, s)

            if v_cnt > 0 and v_cnt < s:
                # 模态特化路径：视觉 / 文本 token 使用各自独立 scale
                x_vis = x[:, :v_cnt, :]
                x_text = x[:, v_cnt:, :]
                x_vis_q = _fake_quant_per_token_symmetric(x_vis / self.vis_scale, bits=self.act_bits) * self.vis_scale
                x_text_q = _fake_quant_per_token_symmetric(x_text / self.text_scale, bits=self.act_bits) * self.text_scale
                x_q = torch.cat([x_vis_q, x_text_q], dim=1)
            else:
                # 退化路径：只有单一模态，使用统一 scale
                x_q = _fake_quant_per_token_symmetric(x / self.scale, bits=self.act_bits) * self.scale
        else:
            x_q = _fake_quant_per_token_symmetric(x / self.scale, bits=self.act_bits) * self.scale

        return nn.functional.linear(x_q, self.weight, self.bias)

    # ---- RMS forward -------------------------------------------------------

    def _forward_rms(self, x: torch.Tensor) -> torch.Tensor:
        """RMS 前向：第一通道 full precision (bf16)，其余通道走 MSQ W4A4 量化。

        分离为两次 matmul:
          y_ch0 = x_ch0 @ W_ch0^T   (bf16 GEMV, overhead 极小)
          y_rest = x_rest_q @ W_rest^T  (W4A4 GEMM)
          y = y_ch0 + y_rest + bias
        """
        x_ch0 = x[..., :1]       # (..., 1)          — full precision
        x_rest = x[..., 1:]      # (..., in_features-1) — to be quantized
        w_ch0 = self.weight[:, :1]    # (out_features, 1)      — full precision
        w_rest = self.weight[:, 1:]   # (out_features, in_features-1) — quantized

        # MSQ 量化 x_rest（逻辑与主 forward 相同，但作用在 ch1: 上）
        if x_rest.dim() == 3:
            b, s, d = x_rest.shape
            v_cnt = _get_visual_token_count(self, default=0)
            v_cnt = min(v_cnt, s)

            if v_cnt > 0 and v_cnt < s:
                x_vis = x_rest[:, :v_cnt, :]
                x_text = x_rest[:, v_cnt:, :]
                x_vis_q = _fake_quant_per_token_symmetric(x_vis / self.vis_scale, bits=self.act_bits) * self.vis_scale
                x_text_q = _fake_quant_per_token_symmetric(x_text / self.text_scale, bits=self.act_bits) * self.text_scale
                x_rest_q = torch.cat([x_vis_q, x_text_q], dim=1)
            else:
                x_rest_q = _fake_quant_per_token_symmetric(x_rest / self.scale, bits=self.act_bits) * self.scale
        else:
            x_rest_q = _fake_quant_per_token_symmetric(x_rest / self.scale, bits=self.act_bits) * self.scale

        y_ch0 = nn.functional.linear(x_ch0, w_ch0)   # bf16 GEMV
        y_rest = nn.functional.linear(x_rest_q, w_rest)  # W4A4 GEMM

        result = y_ch0 + y_rest
        if self.bias is not None:
            result = result + self.bias
        return result


# ---------------------------------------------------------------------------
# 视觉 token 计数辅助函数
# ---------------------------------------------------------------------------

def _get_visual_token_count(module: nn.Module, default: int = 0) -> int:
    """从模型根节点读取当前样本的视觉 token 数量。

    约定：evaluator 在每次 generate 前设置 model._mquant_visual_token_count。
    每个 MQuantLinear 在构造时将 root model weakref 注册到 _mquant_root_registry。
    """
    ref = _mquant_root_registry.get(id(module))
    if ref is not None:
        root = ref()
        if root is not None and hasattr(root, '_mquant_visual_token_count'):
            val = getattr(root, '_mquant_visual_token_count', None)
            if val is not None:
                return int(val)
    return default


# ---------------------------------------------------------------------------
# 递归替换
# ---------------------------------------------------------------------------

def replace_mquant_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    weight_bits: int = 4,
    act_bits: int = 4,
    model_root: nn.Module = None,
    use_rms: bool = False,
    rms_threshold: float = 3.0,
) -> int:
    """递归将 nn.Linear 替换为 MQuantLinear。

    Parameters
    ----------
    model_root: nn.Module or None
        模型根节点引用，MQuantLinear 将在此基础上查找 _mquant_visual_token_count。
        首次调用时应传入根模型（即 evaluator 中 self.model 指向的对象）。
    use_rms: bool
        是否启用 RMS（Rotation Magnitude Suppression）候选。
        实际是否生效取决于 finalize() 时的 auto-detect。
    rms_threshold: float
        RMS auto-detect 阈值：第一通道 absmax > threshold * 其余通道中位数时触发。
    """
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

            mq_layer = MQuantLinear(
                child, weight_bits=weight_bits, act_bits=act_bits,
                model_root=model_root, use_rms=use_rms, rms_threshold=rms_threshold,
            )
            setattr(module, name, mq_layer)
            count += 1
        else:
            count += replace_mquant_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings,
                weight_bits=weight_bits, act_bits=act_bits, model_root=model_root,
                use_rms=use_rms, rms_threshold=rms_threshold,
            )

    return count


# ---------------------------------------------------------------------------
# observe / finalize 入口（供 evaluator 调用）
# ---------------------------------------------------------------------------

def set_mquant_observe(model: nn.Module, enabled: bool) -> int:
    """切换模型内所有 MQuantLinear 的 observe 状态（开启时重置统计量）。"""
    count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, MQuantLinear):
            if enabled:
                sub_module.start_observe()
            else:
                sub_module.observing = False
            count += 1
    return count


def finalize_mquant(model: nn.Module) -> int:
    """对所有 MQuantLinear 计算模态特化 scale 并完成权重量化。"""
    count = 0
    rms_count = 0
    for sub_module in model.modules():
        if isinstance(sub_module, MQuantLinear):
            sub_module.finalize()
            count += 1
            if sub_module._rms_enabled:
                rms_count += 1
    if rms_count > 0:
        print(f"[RMS] {rms_count}/{count} MQuantLinear 层触发 RMS (ch0 full precision)")
    else:
        print(f"[RMS] 0/{count} 层触发 RMS — 第一通道无明显异常（阈值可能偏高或校准数据不够）")
    # reset debug counter for next run
    if hasattr(MQuantLinear, '_rms_debug_count'):
        del MQuantLinear._rms_debug_count
    return count
