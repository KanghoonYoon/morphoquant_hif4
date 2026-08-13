"""
MQuant QuaRot 集成：对 Qwen2.5-Omni 的 Thinker 语言模型骨干做 Hadamard 旋转融合 (R1+R2)。

参考:
- QuaRot (Ashkboos et al., 2024): https://arxiv.org/abs/2404.00456
- MQuant (Yu et al., ACM MM 2025): https://arxiv.org/abs/2502.00425

R1 —— 全局残差流旋转:
  将 hidden_size×hidden_size 的 Hadamard 矩阵融合进所有权重矩阵，使激活分布均匀化。
  对每个线性层: consumer (W @ R) 或 producer (R^T @ W)，与 RMSNorm gamma 吸收交替进行。

R2 —— 注意力 V/O 头内旋转:
  将 head_dim×head_dim 的 Hadamard 矩阵融合进 V projection (producer) 和 O projection
  (consumer) 的每个 head 切片。Q/K 不受影响 (它们走 RoPE)。

性能说明:
  所有旋转操作均在 GPU 上以原始 dtype (bf16/fp16) 完成，不使用 CPU 中转。
  Hadamard 矩阵条件数完美 (κ=1)，bf16 下的舍入误差远小于后续量化误差。
  对于 embedding/lm_head 大矩阵 (151936×2048)，GPU 矩阵乘法 ~0.1 秒 vs CPU ~30 秒。
"""

import torch
import torch.nn as nn

from modules.quarot_layers import make_hadamard_matrix


# ---------------------------------------------------------------------------
# GPU-native 旋转 helpers (替代 quarot_layers.py 中的 CPU 版本)
# ---------------------------------------------------------------------------

def _fold_rmsnorm_gamma_gpu_(norm_module: nn.Module, consumer_linears) -> None:
    """把 RMSNorm 的 gamma 吸收进下游 consumer Linear 的输入维（GPU 原生，保持原始 dtype）。"""
    gamma = norm_module.weight.data  # kept on GPU in original dtype
    for linear in consumer_linears:
        linear.weight.data = linear.weight.data * gamma.unsqueeze(0)  # broadcast mul, stays on GPU
    norm_module.weight.data = torch.ones_like(norm_module.weight.data)


def _apply_consumer_rotation_gpu_(linear: nn.Linear, R: torch.Tensor) -> None:
    """consumer: W_new = W_old @ R（GPU 原生，保持原始 dtype）。"""
    R_cast = R.to(dtype=linear.weight.dtype, device=linear.weight.device)
    linear.weight.data = linear.weight.data @ R_cast


def _apply_producer_rotation_gpu_(linear: nn.Linear, R: torch.Tensor) -> None:
    """producer: W_new = R^T @ W_old（GPU 原生，保持原始 dtype）。"""
    R_cast = R.to(dtype=linear.weight.dtype, device=linear.weight.device)
    linear.weight.data = R_cast.t() @ linear.weight.data


# ---------------------------------------------------------------------------
# R1: 全局残差流旋转
# ---------------------------------------------------------------------------

def apply_quarot_r1_qwen25omni(
    thinker: nn.Module,
    R1: torch.Tensor,
    visual_merger_last: nn.Linear,
    audio_proj: nn.Linear,
) -> None:
    """对 Qwen2.5-Omni Thinker 的 LLM 骨干 + 视觉/音频投影做 R1 旋转融合（GPU 原生）。

    Parameters
    ----------
    thinker: Qwen2_5OmniThinkerForConditionalGeneration
        model.thinker
    R1: Tensor of shape (hidden_size, hidden_size)
        Hadamard 旋转矩阵 (on GPU)
    visual_merger_last: nn.Linear
        thinker.visual.merger.mlp.2 —— 视觉特征投影到 LLM hidden_size
    audio_proj: nn.Linear
        thinker.audio_tower.proj —— 音频特征投影到 LLM hidden_size
    """
    inner = thinker.model  # Qwen2_5OmniThinkerTextModel
    lm_head = thinker.lm_head

    # 1. embedding: E_new = E_old @ R1 (consumer)
    # 直接在 GPU 上以原始 dtype 做矩阵乘法
    emb = inner.embed_tokens
    R1_cast = R1.to(dtype=emb.weight.dtype, device=emb.weight.device)
    emb.weight.data = emb.weight.data @ R1_cast

    # 2. 每层 transformer
    for layer in inner.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        # Attention: input_layernorm → q_proj, k_proj, v_proj (consumers)
        _fold_rmsnorm_gamma_gpu_(layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj])
        _apply_consumer_rotation_gpu_(attn.q_proj, R1)
        _apply_consumer_rotation_gpu_(attn.k_proj, R1)
        _apply_consumer_rotation_gpu_(attn.v_proj, R1)
        # o_proj: producer (output goes back into residual stream)
        _apply_producer_rotation_gpu_(attn.o_proj, R1)

        # FFN: post_attention_layernorm → gate_proj, up_proj (consumers)
        _fold_rmsnorm_gamma_gpu_(layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj])
        _apply_consumer_rotation_gpu_(mlp.gate_proj, R1)
        _apply_consumer_rotation_gpu_(mlp.up_proj, R1)
        # down_proj: producer
        _apply_producer_rotation_gpu_(mlp.down_proj, R1)

    # 3. 最终 norm + lm_head
    _fold_rmsnorm_gamma_gpu_(inner.norm, [lm_head])
    _apply_consumer_rotation_gpu_(lm_head, R1)

    # 4. 视觉投影 (producer: 输出写入 input_embeds 即残差流起点)
    _apply_producer_rotation_gpu_(visual_merger_last, R1)

    # 5. 音频投影 (producer: 同视觉)
    _apply_producer_rotation_gpu_(audio_proj, R1)


# ---------------------------------------------------------------------------
# R2: 注意力 V/O 头内旋转
# ---------------------------------------------------------------------------

def apply_quarot_r2_qwen25omni(
    thinker: nn.Module,
    R2: torch.Tensor,
    num_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 128,
) -> None:
    """对 Qwen2.5-Omni 每层注意力的 V projection (producer) / O projection (consumer)
    做头内 Hadamard 旋转融合。Q/K 不受影响 (RoPE)。(GPU 原生)

    Qwen2.5-Omni attention (q_proj/k_proj/v_proj 分离设计):
      v_proj: Linear(2048, num_kv_heads*head_dim=256)
              → 第 h 个 kv-head 占用行 [h*head_dim : (h+1)*head_dim]
      o_proj: Linear(num_heads*head_dim=2048, 2048)
              → 第 h 个 q-head 占用列 [h*head_dim : (h+1)*head_dim]
    """
    inner = thinker.model

    for layer in inner.layers:
        attn = layer.self_attn

        # V producer: 每 kv-head 的 head_dim 行做 R2^T @ slice
        R2_v = R2.to(dtype=attn.v_proj.weight.dtype, device=attn.v_proj.weight.device)
        w_v = attn.v_proj.weight.data
        for h in range(num_kv_heads):
            r0 = h * head_dim
            r1 = r0 + head_dim
            w_v[r0:r1, :] = R2_v.t() @ w_v[r0:r1, :]

        # O consumer: 每 q-head 的 head_dim 列做 slice @ R2
        R2_o = R2.to(dtype=attn.o_proj.weight.dtype, device=attn.o_proj.weight.device)
        w_o = attn.o_proj.weight.data
        for h in range(num_heads):
            c0 = h * head_dim
            c1 = c0 + head_dim
            w_o[:, c0:c1] = w_o[:, c0:c1] @ R2_o


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def apply_mquant_quarot_to_qwen(model: nn.Module, config) -> None:
    """对 Qwen2.5-Omni 模型应用 MQuant-QuaRot Hadamard 旋转 (R1+R2)，原地修改。

    在 replace_mquant_layers_recursive() 之前调用。
    权重旋转是数学上无损的线性变换，不引入量化误差。
    所有操作在 GPU 上以模型原始 dtype 完成，无需 CPU 中转。
    """
    thinker = model.thinker

    # 从 thinker_config 读取维度信息
    text_config = thinker.model.config
    hidden_size = text_config.hidden_size
    num_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads
    head_dim = hidden_size // num_heads

    # 视觉和音频投影 (都输出到 LLM hidden_size 维残差流)
    visual_merger_last = thinker.visual.merger.mlp[2]
    audio_proj = thinker.audio_tower.proj

    # 随机种子
    seed = getattr(config.quant, "quarot_seed", 2025)
    device = next(thinker.parameters()).device

    # R1: hidden_size × hidden_size
    R1 = make_hadamard_matrix(hidden_size, device=device, seed=seed)
    print(f"[MQuant-QuaRot] 应用 R1 全局残差流旋转 (hidden_size={hidden_size}) [GPU native]...")
    apply_quarot_r1_qwen25omni(thinker, R1, visual_merger_last, audio_proj)
    print("[MQuant-QuaRot] R1 完成。")

    # R2: head_dim × head_dim
    if getattr(config.quant, "quarot_rotate_v_o", True):
        R2 = make_hadamard_matrix(head_dim, device=device, seed=None if seed is None else seed + 1)
        print(f"[MQuant-QuaRot] 应用 R2 注意力 V/O 头内旋转 (head_dim={head_dim}, "
              f"num_heads={num_heads}, num_kv_heads={num_kv_heads}) [GPU native]...")
        apply_quarot_r2_qwen25omni(
            thinker, R2, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        print("[MQuant-QuaRot] R2 完成。")

    print("[MQuant-QuaRot] Hadamard 旋转全部完成。")
