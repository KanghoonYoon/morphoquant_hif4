"""
QuaRot (Ashkboos et al., 2024) 旋转融合实现，适配 InternVL2.5 (InternLM2 语言模型骨干)。

范围：
  - R1: 全局残差流旋转 (embedding / output / wqkv / wo / w1 / w3 / w2 / mlp1 最后一层 Linear)
  - R2: 注意力 V/O 头内旋转 (head_dim 大小的 Hadamard，仅作用于 V 的输出切片和 O 的输入切片)
  - 不包含 MLP down_proj 输入侧的在线 Hadamard 变换 (intermediate_size 非 2 的幂，跳过，留作后续工作)

R1/R2 都是纯离线权重融合操作 (一次性矩阵乘法改写权重)，在数学上不引入误差、不产生额外推理开销。
真正的精度损失来自之后对旋转后权重做的 RTN 伪量化。

权重量化复用 modules/rtn_layers.py 的逐输出通道 RTN 方案；但激活量化必须是**逐 token**
(而不是 modules/rtn_layers.py 里现成的逐张量) ——QuaRot 的旋转只让单个 token 内部的
hidden_size 维激活分布变得均匀，并不会拉齐不同 token (尤其是 InternVL 的图像 token 与文本
token) 之间的整体幅值；逐张量量化会被序列里任意一个幅值较大的 token 主导 scale，
导致其余 token 精度被压垮、生成乱码。因此本文件单独实现 QuaRotRTNLinear，不复用/不修改
共享的 RTNQuantLinear，避免影响 Qwen 端已有的逐张量量化配置。
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


def _fake_quant_per_token_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """逐 token (最后一维 hidden_size 各自一个 scale) 的对称伪量化。"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return x_q * scale


class QuaRotRTNLinear(nn.Module):
    """QuaRot 专用 RTN 伪量化 Linear：权重逐输出通道伪量化 + 可选激活逐 token 动态伪量化。"""

    def __init__(self, original_linear: nn.Linear, weight_bits: int = 4, act_bits=4):
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
            x = _fake_quant_per_token_symmetric(x, bits=self.act_bits)

        return nn.functional.linear(x, self.weight, self.bias)


def replace_quarot_rtn_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    weight_bits: int = 4,
    act_bits=4,
) -> int:
    """递归将 nn.Linear 替换为 QuaRotRTNLinear (逐 token 激活量化)。"""
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue

            q_layer = QuaRotRTNLinear(child, weight_bits=weight_bits, act_bits=act_bits)
            setattr(module, name, q_layer)
            count += 1
        else:
            count += replace_quarot_rtn_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings,
                weight_bits=weight_bits, act_bits=act_bits,
            )

    return count


def make_hadamard_matrix(n: int, device=None, dtype=torch.float32, seed=None) -> torch.Tensor:
    """构造 n x n (n 为 2 的幂) 正交 Hadamard 矩阵，可选叠加随机 ±1 对角符号翻转。"""
    if n & (n - 1) != 0:
        raise ValueError(f"make_hadamard_matrix requires n to be a power of 2, got {n}")

    h = torch.tensor([[1.0]], dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1),
        ], dim=0)

    h = h / (n ** 0.5)

    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (n,), generator=gen, dtype=dtype) * 2 - 1
        h = h * signs.unsqueeze(0)

    return h.to(device=device)


def _fold_rmsnorm_gamma_(norm_module: nn.Module, consumer_linears) -> None:
    """把 RMSNorm 的可学习 weight (gamma) 吸收进下游 consumer Linear 的输入维，norm.weight 重置为 1。

    在 CPU 上以 float32 完成矩阵运算，避免大词表 embedding/output 层的 fp64 临时张量
    在 GPU 上引发显存溢出 (该操作只在模型加载时执行一次，CPU 算力/内存代价可忽略)。
    """
    gamma = norm_module.weight.data.to("cpu", torch.float32)
    for linear in consumer_linears:
        orig_device, orig_dtype = linear.weight.device, linear.weight.dtype
        w = linear.weight.data.to("cpu", torch.float32)
        linear.weight.data = (w * gamma.unsqueeze(0)).to(orig_device, orig_dtype)
    norm_module.weight.data = torch.ones_like(norm_module.weight.data)


def _apply_consumer_rotation_(linear: nn.Linear, R: torch.Tensor) -> None:
    """consumer: 输入是被旋转的残差流，输出不进残差流。W_new = W_old @ R (沿 in 维右乘)。"""
    orig_device, orig_dtype = linear.weight.device, linear.weight.dtype
    w = linear.weight.data.to("cpu", torch.float32)
    R_cpu = R.to("cpu", torch.float32)
    linear.weight.data = (w @ R_cpu).to(orig_device, orig_dtype)


def _apply_producer_rotation_(linear: nn.Linear, R: torch.Tensor) -> None:
    """producer: 输出要并入被旋转的残差流。W_new = R^T @ W_old (沿 out 维左乘)。"""
    orig_device, orig_dtype = linear.weight.device, linear.weight.dtype
    w = linear.weight.data.to("cpu", torch.float32)
    R_cpu = R.to("cpu", torch.float32)
    linear.weight.data = (R_cpu.t() @ w).to(orig_device, orig_dtype)


def apply_quarot_r1_(language_model: nn.Module, mlp1_last_linear: nn.Linear, R1: torch.Tensor) -> None:
    """对 InternLM2 语言模型骨干 + InternVL mlp1 投影层做 R1 全局残差流旋转融合。"""
    inner = language_model.model  # InternLM2Model
    output_linear = language_model.output  # 词表投影, 名为 output 而非 lm_head

    # embedding: E_new = E_old @ R1
    emb = inner.tok_embeddings
    orig_device, orig_dtype = emb.weight.device, emb.weight.dtype
    emb_w = emb.weight.data.to("cpu", torch.float32)
    R1_cpu = R1.to("cpu", torch.float32)
    emb.weight.data = (emb_w @ R1_cpu).to(orig_device, orig_dtype)

    for layer in inner.layers:
        attn = layer.attention
        ffn = layer.feed_forward

        _fold_rmsnorm_gamma_(layer.attention_norm, [attn.wqkv])
        _apply_consumer_rotation_(attn.wqkv, R1)
        _apply_producer_rotation_(attn.wo, R1)

        _fold_rmsnorm_gamma_(layer.ffn_norm, [ffn.w1, ffn.w3])
        _apply_consumer_rotation_(ffn.w1, R1)
        _apply_consumer_rotation_(ffn.w3, R1)
        _apply_producer_rotation_(ffn.w2, R1)

    # 最终 norm 的 gamma 吸收进 output 投影，再对 output 做 consumer 旋转
    _fold_rmsnorm_gamma_(inner.norm, [output_linear])
    _apply_consumer_rotation_(output_linear, R1)

    # mlp1 (视觉->语言投影) 的最后一层 Linear 输出直接写入 input_embeds，需同坐标系，按 producer 处理
    _apply_producer_rotation_(mlp1_last_linear, R1)


def apply_quarot_r2_(
    language_model: nn.Module,
    R2: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> None:
    """对每层注意力的 V (wqkv 输出切片) / O (wo 输入切片) 做头内旋转融合，Q/K/RoPE 不受影响。

    InternLM2 的 wqkv 输出按 kv-head 分组交织排列 (见 modeling_internlm2.py 的
    `rearrange('b q (h gs d) -> b q h gs d', gs=2+num_key_value_groups)`)：每个 kv-head 占
    `(num_key_value_groups+2)*head_dim` 行，组内前 `num_key_value_groups` 个 head_dim 块是该组
    共享的 Q heads，倒数第二块是 K，最后一块才是 V —— V 并不是整块矩阵末尾的连续 1024 行。
    """
    inner = language_model.model
    R2_cpu = R2.to("cpu", torch.float32)

    num_key_value_groups = num_heads // num_kv_heads
    group_size = (num_key_value_groups + 2) * head_dim

    for layer in inner.layers:
        attn = layer.attention
        wqkv = attn.wqkv
        wo = attn.wo

        # V 切片: 每个 kv-head 分组内最后一个 head_dim 块，逐块 producer 旋转
        orig_device, orig_dtype = wqkv.weight.device, wqkv.weight.dtype
        w_full = wqkv.weight.data.to("cpu", torch.float32)
        for h in range(num_kv_heads):
            group_start = h * group_size
            r0 = group_start + (num_key_value_groups + 1) * head_dim
            r1 = r0 + head_dim
            w_full[r0:r1, :] = R2_cpu.t() @ w_full[r0:r1, :]
        wqkv.weight.data = w_full.to(orig_device, orig_dtype)

        # O 输入切片: wo 输入的每个 128 列块 (对应每个 query head)，逐块 consumer 旋转，所有 head 共用同一个 R2
        orig_device, orig_dtype = wo.weight.device, wo.weight.dtype
        w_o = wo.weight.data.to("cpu", torch.float32)
        for h in range(num_heads):
            c0 = h * head_dim
            c1 = c0 + head_dim
            w_o[:, c0:c1] = w_o[:, c0:c1] @ R2_cpu
        wo.weight.data = w_o.to(orig_device, orig_dtype)


def apply_quarot_to_internvl(model: nn.Module, config) -> None:
    """顶层入口：对 InternVL2.5 的语言模型骨干 (+ mlp1 投影层) 应用 QuaRot R1(+R2) 旋转融合，原地修改。"""
    language_model = model.language_model
    hidden_size = language_model.config.hidden_size
    num_heads = language_model.config.num_attention_heads
    num_kv_heads = language_model.config.num_key_value_heads
    head_dim = hidden_size // num_heads

    seed = getattr(config.quant, "quarot_seed", 2025)
    device = next(language_model.parameters()).device

    R1 = make_hadamard_matrix(hidden_size, device=device, seed=seed)
    apply_quarot_r1_(language_model, model.mlp1[3], R1)

    if getattr(config.quant, "quarot_rotate_v_o", True):
        R2 = make_hadamard_matrix(head_dim, device=device, seed=None if seed is None else seed + 1)
        apply_quarot_r2_(language_model, R2, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
