import torch
import torch.nn as nn
import sys

# Ensure quant_cy is in the path
sys.path.append("/private/wy/MorphoQuant/modules/HiF8_NPU_GPU_simulator/GPU_Cuda/hif8_cuda")

class HiF8Linear(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 复制原本的参数 (保持 BF16/FP16)
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
            
        self.weight_quantized = False
        self.quant_config_str = 'hif8'

    def quantize_weight(self):
        """对权重进行一次性模拟量化 (In-place)"""
        if not self.weight_quantized:
            # 确保在 CUDA 上，且 quant_cy 可用
            if not self.weight.is_cuda:
                return 
            
            try:
                from quant_cy import quant_dequant_float, QType
                # 模拟权重: FP -> HiF8 -> FP
                # 使用 QType(str).dim(0) 通常指 Per-Channel? 或者默认配置
                # 依据 hif8.py 示例使用
                self.weight.data = quant_dequant_float(
                    self.weight.data, 
                    QType(self.quant_config_str).dim(0), 
                    force_fp32=True
                )
                self.weight_quantized = True
            except ImportError:
                print("Error: quant_cy module not found. HiF8 simulation failed.")
            except Exception as e:
                print(f"Error during weight quantization: {e}")

    def forward(self, x):
        # 1. 确保权重已量化 (Lazy Init)
        if not self.weight_quantized:
            if self.weight.device != x.device:
                self.weight.data = self.weight.data.to(x.device)
            self.quantize_weight()

        # 2. 模拟输入激活量化: FP -> HiF8 -> FP
        # x shape: [B, L, D]
        # quant_dequant_float 要求输入连续且在 CUDA
        from quant_cy import quant_dequant_float, QType
        
        x_quant = quant_dequant_float(
            x.contiguous(), 
            QType(self.quant_config_str).dim(0), 
            force_fp32=True
        )
        
        # 3. 计算 (使用模拟后的 dirty weight 和 dirty input)
        return nn.functional.linear(x_quant, self.weight, self.bias)

class MorphoHiF8Linear(nn.Module):
    def __init__(self, original_linear, config, name):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.layer_name = name

        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
            
        self.weight_quantized = False
        self.quant_config_str = 'hif8'

        # 引入 MorphoQuant 所属的激活量化器
        from bitsandbytes.quantization_utils.quant_modules import QuantAct
        
        # 这些参数是为了适应原始 QuantAct 对层计数、LLM 特征等判定的需求
        # 这里简单适配一下参数
        llama_layer = True
        if self.in_features in (1024,):  # mm_projector 等简化判断
            llama_layer = False
            
        self.quant_activation = QuantAct(
            activation_bit=config.quant.activation_bitwidth,
            input_dim=self.in_features,
            llama_layer=llama_layer,
            count_block=1,  # 简化版本，可能需要针对网络结构调整
            count_layer=1
        )
        self.quant_activation.set_gamma(gamma_inf=config.quant.gamma_inf, gamma_cos=config.quant.gamma_cos)
        self.quant_activation.set_lp_norm(lp_norm=config.quant.lp_norm)
        self.quant_activation.set_cosine_loss(use_cosine_loss=config.quant.use_cosine_loss)
        self.quant_activation.layer_name = name
        
        # 标志：我们是否拦截 QuantAct 里面原生的整形(Int)截断操作，换成 HiF8 
        # 我们用一个 hook 或 wrapper 手动截断/重写。为了禁止改动底层，我们选择对原本 QuantAct 的量化结果直接返回，
        # 或者自己执行 QuantAct 算好的缩放之后送入 HiF8。
        # 由于这里我们要求 Beta Path：利用它的缩放因子 -> 平滑 -> HiF8伪量化，
        # 但实际上 QuantAct 自身直接输出了 fake-quant 张量，我们可以选择直接拿它的输入经过 HiF8处理，
        # 或者如果你想更深度结合，可以 monkey patch 它内部的 forward。这里选择了最外层的平滑输入代理。

    def quantize_weight(self):
        if not self.weight_quantized:
            if not self.weight.is_cuda:
                return 
            try:
                from quant_cy import quant_dequant_float, QType
                self.weight.data = quant_dequant_float(
                    self.weight.data, 
                    QType(self.quant_config_str).dim(0), 
                    force_fp32=True
                )
                self.weight_quantized = True
            except ImportError:
                print("Error: quant_cy module not found. HiF8 simulation failed.")

    def _hif8_quantize_activation(self, x):
        from quant_cy import quant_dequant_float, QType
        return quant_dequant_float(
            x.contiguous(),
            QType(self.quant_config_str).dim(0),
            force_fp32=True
        )

    def _build_sparse_mask(self, x):
        outlier_mask = getattr(self.quant_activation, 'outlier_mask', None)
        if outlier_mask is None:
            return None

        if hasattr(self.quant_activation, 'compensation_limit'):
            limit = self.quant_activation.compensation_limit
        elif hasattr(self.quant_activation, '_temp_best_min') and hasattr(self.quant_activation, '_temp_best_max'):
            sparse_buffer_ratio = getattr(self.quant_activation, 'sparse_buffer_ratio', 0.8)
            limit = torch.max(
                self.quant_activation._temp_best_max.abs(),
                self.quant_activation._temp_best_min.abs(),
            ) * sparse_buffer_ratio
        elif hasattr(self.quant_activation, 'activation_range_min') and hasattr(self.quant_activation, 'activation_range_max'):
            sparse_buffer_ratio = getattr(self.quant_activation, 'sparse_buffer_ratio', 0.8)
            limit = torch.max(
                self.quant_activation.activation_range_max.abs(),
                self.quant_activation.activation_range_min.abs(),
            ) * sparse_buffer_ratio
        else:
            return None

        limit = limit.to(x.device).to(x.dtype)
        channel_mask = outlier_mask.to(x.device).bool()

        if limit.dim() == 1:
            if x.shape[-1] == limit.numel():
                view_shape = [1] * (x.ndim - 1) + [-1]
            elif x.ndim >= 2 and x.shape[-2] == limit.numel():
                view_shape = [1] * x.ndim
                view_shape[-2] = -1
            else:
                view_shape = [1] * (x.ndim - 1) + [-1]

            limit = limit.view(*view_shape)
            channel_mask = channel_mask.view(*view_shape)

        return (x.abs() > limit) & channel_mask

    def forward(self, x):
        if not self.weight_quantized:
            if self.weight.device != x.device:
                self.weight.data = self.weight.data.to(x.device)
            self.quantize_weight()

        # Keep MorphoQuant's calibration/search state updates, but do not feed
        # its int-like fake-quant output into HiF8 again.
        if getattr(self.quant_activation, '_calibrate', False) or getattr(self.quant_activation, 'search', False):
            _ = self.quant_activation(x)

        sparse_mask = self._build_sparse_mask(x)
        if sparse_mask is None:
            x_main = self._hif8_quantize_activation(x)
            return nn.functional.linear(x_main, self.weight, self.bias)

        x_main = torch.where(sparse_mask, torch.zeros_like(x), x)
        x_sparse = torch.where(sparse_mask, x, torch.zeros_like(x))
        x_main = self._hif8_quantize_activation(x_main)

        out_main = nn.functional.linear(x_main, self.weight, self.bias)
        out_sparse = nn.functional.linear(x_sparse, self.weight, None)
        return out_main + out_sparse

def _should_skip_module(fullname, skip_substrings):
    return any(substring in fullname for substring in (skip_substrings or ()))


def replace_morpho_hif8_layers_recursive(module, config, prefix=""):
    """供 Morpho + HiF8 使用的全新递归替换"""
    skip_substrings = getattr(config.quant, "skip_module_substrings", None)
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name: 
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue
                
            morpho_layer = MorphoHiF8Linear(child, config, fullname)
            setattr(module, name, morpho_layer)
            count += 1
        else:
            count += replace_morpho_hif8_layers_recursive(child, config, prefix=fullname)
            
    return count

def replace_hif8_layers_recursive(module, prefix="", skip_substrings=None):
    """递归替换 Linear 为 HiF8Linear"""
    count = 0
    # 遍历所有子模块 (immediate children)
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear):
            # 过滤策略: 跳过 LM Head, Embedding 等
            if "lm_head" in name or "output" in name: 
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue
                
            # 执行替换
            hif8_layer = HiF8Linear(child)
            setattr(module, name, hif8_layer)
            count += 1
        else:
            # 递归
            count += replace_hif8_layers_recursive(child, prefix=fullname, skip_substrings=skip_substrings)
            
    return count
