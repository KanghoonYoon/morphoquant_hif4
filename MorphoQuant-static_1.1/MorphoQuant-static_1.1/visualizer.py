import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Global variable to store visual token range for plotting
last_visual_range = None

# ========== 捕获激活分布的 Hook ==========
# 保存每层输入激活的统计与样本
layer_stats = defaultdict(list)
layer_inputs = defaultdict(list)

# 定义 hook，用于捕获输入分布
def activation_stats_hook(module, input, output):
    x = input[0].detach().cpu()
    v_len = getattr(module, "visual_token_length", 0)
    if x.ndim == 3 and v_len > 0 and x.shape[1] > v_len:
        visual_tokens = x[:, :v_len, :].flatten()
        text_tokens = x[:, v_len:, :].flatten()
        for label, data in [("visual", visual_tokens), ("text", text_tokens)]:
            if data.numel() > 100000:
                data = data[torch.randint(0, data.numel(), (10000,))]
            layer_inputs[f"{module.layer_name}_{label}"].append(data)
            layer_stats[f"{module.layer_name}_{label}"].append({
                "mean": data.mean().item(),
                "std": data.std().item(),
                "min": data.min().item(),
                "max": data.max().item(),
            })
    else:
        # fallback（无视觉输入或单模态）
        x = x.flatten()
        if x.numel() > 100000:
            x = x[torch.randint(0, x.numel(), (10000,))]
        layer_inputs[module.layer_name].append(x)
        layer_stats[module.layer_name].append({
            "mean": x.mean().item(),
            "std": x.std().item(),
            "min": x.min().item(),
            "max": x.max().item(),
        })

# 注册 hook 时附加视觉 token 长度信息
def register_activation_hooks(model, v_len=None):
    hooks = []
    for name, module in model.named_modules():
        # 适配 HiF8 或 Morpho 结合版本的类名
        is_linear_target = isinstance(module, nn.Linear) 
        if hasattr(module, 'quant_config_str') and hasattr(module, 'weight_quantized'):
            # 这是自定义的 HiF8 或 MorphoHiF8 包装层
            is_linear_target = True

        if "thinker.model.layers" in name and is_linear_target:
            module.layer_name = name
            if v_len is not None:
                module.visual_token_length = v_len
            hook = module.register_forward_hook(activation_stats_hook)
            hooks.append(hook)
    return hooks

def plot_activation_histograms(save_dir="/private/wy/logs/MorphoQuant/activation_histograms"):
    """
    绘制学术级的激活值直方图，重点突出长尾分布。
    """
    os.makedirs(save_dir, exist_ok=True)

    # === 1. 全局学术风格设置 ===
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 16,
        "axes.linewidth": 2,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": '--',
    })

    for layer_name, tensors in layer_inputs.items():
        # 展平数据，确保是一维的
        if isinstance(tensors, list):
            data = torch.cat(tensors).flatten().float().numpy()
        else:
            data = tensors.flatten().float().numpy()

        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)

        # 创建画布 (4:3 比例)
        plt.figure(figsize=(8, 6))

        # === 2. 绘制直方图 (关键：使用 log=True) ===
        # 增加 bins 数量以获得更细腻的分布
        plt.hist(data, bins=150, log=True, 
                 color="#2c3e50", edgecolor='none', alpha=0.85)
        
        plt.xlabel("Activation Magnitude", fontsize=18)
        plt.ylabel("Frequency (Log Scale)", fontsize=18)

        # 标记极值线
        plt.axvline(min_val, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
        plt.axvline(max_val, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)

        # 添加漂亮的统计信息框
        stats_text = (
            f"$\\bf{{Statistics}}$\n"
            f"Min: {min_val:.2f}\n"
            f"Max: {max_val:.2f}\n"
            f"$\\mu$: {mean_val:.2f}\n"
            f"$\\sigma$: {std_val:.2f}"
        )
        
        plt.gca().text(0.95, 0.95, stats_text,
                       transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7'))

        if max_val > mean_val + 5 * std_val:
            plt.annotate('Heavy Tail\n(Outliers)', 
                         xy=(max_val, 1), xytext=(max_val * 0.7, 10),
                         arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2),
                         fontsize=14, color='red', ha='right')
        if min_val < mean_val - 5 * std_val:
            plt.annotate('Long Tail\n(Outliers)', 
                         xy=(min_val, 1), xytext=(min_val * 0.7, 10),
                         arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2),
                         fontsize=14, color='red', ha='left', fontweight='bold')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", layer_name)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{safe_name}.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150)
        plt.close()

    print(f"✅ High-quality activation histograms saved in: {save_dir}")

# ========== Attention Map Hook ==========
attention_maps = defaultdict(list)

def attention_hook(module, input, output):
    if isinstance(output, tuple) and len(output) > 1:
        attn_weights = output[1]
        if attn_weights is not None:
            attention_maps[module.layer_name].append(attn_weights.detach().cpu())

def register_attention_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if "self_attn" in name:
            module.layer_name = name
            hook = module.register_forward_hook(attention_hook)
            hooks.append(hook)
    return hooks

def plot_and_save_attention_maps(pid, save_dir="attention_maps", save_per_head_attention=False):
    os.makedirs(save_dir, exist_ok=True)
    tensor_save_dir = os.path.join(save_dir, "tensors")
    os.makedirs(tensor_save_dir, exist_ok=True)
    
    for layer_name, step_outputs in attention_maps.items():
        if not step_outputs:
            continue
        
        prefill_tensor = step_outputs[0]
        final_decode_tensor = step_outputs[-1]
        
        num_heads = prefill_tensor.shape[1]
        seq_prefill = prefill_tensor.shape[2]
        seq_total = final_decode_tensor.shape[-1]
        
        full_attn_heads = torch.zeros((num_heads, seq_total, seq_total), dtype=torch.float)
        full_attn_heads[:, :seq_prefill, :seq_prefill] = prefill_tensor[0].float().cpu()
        
        if len(step_outputs) > 1:
            current_row = seq_prefill
            for i in range(1, len(step_outputs)):
                step_tensor = step_outputs[i]
                seq_len_curr = step_tensor.shape[-1]
                full_attn_heads[:, current_row, :seq_len_curr] = step_tensor[0, :, 0, :].float().cpu()
                current_row += 1
        
        def plot_matrix(matrix, title_suffix, filename_suffix, output_dir):
            plt.figure(figsize=(12, 10))
            plt.imshow(matrix.numpy(), cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar()
            
            plt.axhline(y=seq_prefill-0.5, color='r', linestyle='--', linewidth=1, alpha=0.5)
            plt.axvline(x=seq_prefill-0.5, color='r', linestyle='--', linewidth=1, alpha=0.5)
            
            if last_visual_range:
                for start, end in last_visual_range:
                    color = 'white'
                    lw = 1.5
                    alpha = 0.8
                    plt.axhline(y=start-0.5, color=color, linestyle='-', linewidth=lw, alpha=alpha)
                    plt.axhline(y=end+0.5, color=color, linestyle='-', linewidth=lw, alpha=alpha)
                    plt.axvline(x=start-0.5, color=color, linestyle='-', linewidth=lw, alpha=alpha)
                    plt.axvline(x=end+0.5, color=color, linestyle='-', linewidth=lw, alpha=alpha)

            plt.title(f"Attention Map - {layer_name}\nPID: {pid} {title_suffix}")
            plt.xlabel("Key Token Index")
            plt.ylabel("Query Token Index")
            
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", layer_name)
            plt.savefig(os.path.join(output_dir, f"{pid}_{safe_name}{filename_suffix}.png"))
            plt.close()

        avg_attn = full_attn_heads.mean(dim=0)
        
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", layer_name)
        tensor_path = os.path.join(tensor_save_dir, f"{pid}_{safe_name}.pt")
        torch.save(avg_attn, tensor_path)

        plot_matrix(avg_attn, "(Average)", "_avg", save_dir)
        
        if save_per_head_attention:
            safe_layer_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', layer_name)
            head_dir = os.path.join(save_dir, f"{pid}_{safe_layer_name}_heads")
            os.makedirs(head_dir, exist_ok=True)
            
            for h in range(num_heads):
                plot_matrix(full_attn_heads[h], f"(Head {h})", f"_head_{h}", head_dir)
    
    attention_maps.clear()
