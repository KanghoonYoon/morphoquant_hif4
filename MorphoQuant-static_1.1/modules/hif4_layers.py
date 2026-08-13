"""
4-bit 浮点量化层实现（HiF4 / NVFP4）。

提供两种模式:
  1. Pure FP4   — 直接将 nn.Linear 替换为 HiF4Linear（模拟 FP→FP4→FP 伪量化）
  2. Morpho+FP4 — 融合 MorphoQuant 的激活缩放与 FP4 伪量化（Beta 路径）

具体的 4-bit 格式由 YAML 里的 `quant.fp4_qtype` 决定:
  - `hifx4` (默认) — HiFloat4
  - `nvfp4`        — NVFP4 (E4M3 per-16-block scale + E2M1)

依赖:
  - modules/HiFloat4-main/hif4_gpu/quant_cy （需先编译 CUDA 内核: cd quant_cy/base/cusrc && python setup.py build_ext --inplace）
"""

import importlib
import os
import sys

import torch
import torch.nn as nn

from modules.device_utils import get_device_type, on_accelerator

# ---- FP4 模拟器后端 (CUDA -> quant_cy / Ascend NPU -> quant_cy_npu) ----
# 两个后端导出完全相同的 API (quant_dequant_float / QType)，且都支持 hifx4 与 nvf4。
_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_FP4_BACKENDS = {
    # device_type: (相对 modules/ 的模拟器根目录, 包名)
    "cuda": (os.path.join("HiFloat4-main", "hif4_gpu"), "quant_cy"),
    "npu": (os.path.join("HiFloat4-main", "hifx4_npu"), "quant_cy_npu"),
}
# 旧的绝对路径保留为兜底，兼容已部署在 /private/wy/MorphoQuant 的机器
_LEGACY_MODULES_DIR = "/private/wy/MorphoQuant/modules"

_QUANT_CY = None


def load_quant_cy():
    """按当前设备加载 FP4 模拟器后端并返回模块对象（结果缓存）。"""
    global _QUANT_CY
    if _QUANT_CY is not None:
        return _QUANT_CY

    device = get_device_type()
    if device not in _FP4_BACKENDS:
        raise RuntimeError(
            f"FP4 simulation needs a CUDA GPU or an Ascend NPU, but the detected device is {device!r}."
        )

    subdir, pkg_name = _FP4_BACKENDS[device]
    for root in (_MODULES_DIR, _LEGACY_MODULES_DIR):
        path = os.path.join(root, subdir)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    _QUANT_CY = importlib.import_module(pkg_name)
    return _QUANT_CY


def is_npu_backend() -> bool:
    return get_device_type() == "npu"


def _backend_build_hint() -> str:
    """给出当前设备对应后端的编译命令，便于排错。"""
    subdir, pkg_name = _FP4_BACKENDS.get(get_device_type(), _FP4_BACKENDS["cuda"])
    root = subdir.replace(os.sep, "/")
    return f"cd modules/{root}/{pkg_name}/base/cusrc && python setup.py build_ext --inplace"


# ---- 4-bit 浮点格式解析 ----
# key 是 YAML 里可写的名字，value 是 quant_cy QType 能识别的描述串。
_FP4_QTYPE_ALIASES = {
    "hif4": "hifx4",
    "hifx4": "hifx4",
    "nvf4": "nvf4",
    "nvfp4": "nvf4",
}

DEFAULT_FP4_QTYPE = "hifx4"


def resolve_fp4_qtype(qtype) -> str:
    """把 YAML 里的 dtype 名称归一化为 quant_cy 的 QType 描述串。

    未知名称直接报错，避免拼写错误静默回退到 hifx4 而污染实验结果。
    """
    if qtype is None:
        return DEFAULT_FP4_QTYPE
    key = str(qtype).strip().lower()
    if key not in _FP4_QTYPE_ALIASES:
        raise ValueError(
            f"Unsupported quant.fp4_qtype {qtype!r}; "
            f"expected one of {sorted(_FP4_QTYPE_ALIASES)}."
        )
    return _FP4_QTYPE_ALIASES[key]


def resolve_fp4_qtype_from_config(config) -> str:
    """从 AppConfig 读取并归一化 fp4_qtype。"""
    return resolve_fp4_qtype(getattr(config.quant, "fp4_qtype", DEFAULT_FP4_QTYPE))


class HiF4Linear(nn.Module):
    """纯 4-bit 浮点伪量化 Linear 层 (HiF4 / NVFP4)。

    行为:
      - 首次 forward 时对权重执行一次性伪量化（原地修改 self.weight）
      - 每次 forward 对输入激活执行伪量化
      - 使用 quant_dequant_float(..., QType(qtype).dim(0), force_fp32=True)
    """

    def __init__(self, original_linear: nn.Linear, qtype: str = DEFAULT_FP4_QTYPE):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # 复制原始权重/偏置（保持 BF16/FP16）
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.weight_quantized = False
        self.quant_config_str = resolve_fp4_qtype(qtype)

    def quantize_weight(self):
        """对权重执行一次性伪量化 (In-place)。"""
        if self.weight_quantized:
            return
        if not on_accelerator(self.weight):
            return

        try:
            qc = load_quant_cy()

            self.weight.data = qc.quant_dequant_float(
                self.weight.data,
                qc.QType(self.quant_config_str).dim(0),
                force_fp32=False,
            )
            self.weight_quantized = True
        except ImportError:
            print("Error: FP4 simulator backend not found. "
                  f"Please build: {_backend_build_hint()}")
        except Exception as e:
            print(f"Error during FP4 weight quantization: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Lazy init: 确保权重已量化
        if not self.weight_quantized:
            if self.weight.device != x.device:
                self.weight.data = self.weight.data.to(x.device)
            self.quantize_weight()

        # 2. 输入激活 FP4 伪量化
        qc = load_quant_cy()

        x_quant = qc.quant_dequant_float(
            x.contiguous(),
            qc.QType(self.quant_config_str).dim(-1),
            force_fp32=True,
        )

        # 3. 使用量化后的权重和输入做线性变换
        return nn.functional.linear(x_quant, self.weight, self.bias)


class MorphoHiF4Linear(nn.Module):
    """MorphoQuant + 4-bit 浮点 (HiF4 / NVFP4) 融合 Linear 层（Beta 路径）。

    与 MorphoHiF8Linear 对应:
      - 保留 MorphoQuant 的 QuantAct 缩放/搜索状态更新
      - 将 int-like fake-quant 替换为 FP4 伪量化（格式由 config.quant.fp4_qtype 决定）
      - 支持 channel-wise outlier mask 的 sparse+dense 双路径
    """

    def __init__(self, original_linear: nn.Linear, config, name: str):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.layer_name = name

        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.weight_quantized = False
        self.quant_config_str = resolve_fp4_qtype_from_config(config)

        # 引入 MorphoQuant 的激活量化器
        from bitsandbytes.quantization_utils.quant_modules import QuantAct

        llama_layer = True
        if self.in_features in (1024,):
            llama_layer = False

        self.quant_activation = QuantAct(
            activation_bit=config.quant.activation_bitwidth,
            input_dim=self.in_features,
            llama_layer=llama_layer,
            count_block=1,
            count_layer=1,
        )
        self.quant_activation.set_gamma(
            gamma_inf=config.quant.gamma_inf,
            gamma_cos=config.quant.gamma_cos,
        )
        self.quant_activation.set_lp_norm(lp_norm=config.quant.lp_norm)
        self.quant_activation.set_cosine_loss(
            use_cosine_loss=config.quant.use_cosine_loss,
        )
        self.quant_activation.layer_name = name

    # ---- Weight quantization ----
    def quantize_weight(self):
        if self.weight_quantized:
            return
        if not on_accelerator(self.weight):
            return
        try:
            qc = load_quant_cy()

            self.weight.data = qc.quant_dequant_float(
                self.weight.data,
                qc.QType(self.quant_config_str).dim(0),
                force_fp32=True,
            )
            self.weight_quantized = True
        except ImportError:
            print("Error: FP4 simulator backend not found. "
                  f"Please build: {_backend_build_hint()}")
        except Exception as e:
            print(f"Error during FP4 weight quantization: {e}")

    # ---- Activation quantization ----
    def _hif4_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        qc = load_quant_cy()

        return qc.quant_dequant_float(
            x.contiguous(),
            qc.QType(self.quant_config_str).dim(-1),
            force_fp32=True,
        )

    # ---- Sparse mask (from MorphoQuant's outlier detection) ----
    def _build_sparse_mask(self, x: torch.Tensor):
        outlier_mask = getattr(self.quant_activation, "outlier_mask", None)
        if outlier_mask is None:
            return None

        if hasattr(self.quant_activation, "compensation_limit"):
            limit = self.quant_activation.compensation_limit
        elif hasattr(self.quant_activation, "_temp_best_min") and hasattr(
            self.quant_activation, "_temp_best_max"
        ):
            sparse_buffer_ratio = getattr(
                self.quant_activation, "sparse_buffer_ratio", 0.8
            )
            limit = torch.max(
                self.quant_activation._temp_best_max.abs(),
                self.quant_activation._temp_best_min.abs(),
            ) * sparse_buffer_ratio
        elif hasattr(self.quant_activation, "activation_range_min") and hasattr(
            self.quant_activation, "activation_range_max"
        ):
            sparse_buffer_ratio = getattr(
                self.quant_activation, "sparse_buffer_ratio", 0.8
            )
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

    # ---- Forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.weight_quantized:
            if self.weight.device != x.device:
                self.weight.data = self.weight.data.to(x.device)
            self.quantize_weight()

        # 保持 MorphoQuant 的校准/搜索状态更新，但不将其 int-like 输出喂入 HiF4
        if getattr(self.quant_activation, "_calibrate", False) or getattr(
            self.quant_activation, "search", False
        ):
            _ = self.quant_activation(x)

        sparse_mask = self._build_sparse_mask(x)
        if sparse_mask is None:
            x_main = self._hif4_quantize_activation(x)
            return nn.functional.linear(x_main, self.weight, self.bias)

        # Sparse + Dense 双路径
        x_main = torch.where(sparse_mask, torch.zeros_like(x), x)
        x_sparse = torch.where(sparse_mask, x, torch.zeros_like(x))
        x_main = self._hif4_quantize_activation(x_main)

        out_main = nn.functional.linear(x_main, self.weight, self.bias)
        out_sparse = nn.functional.linear(x_sparse, self.weight, None)
        return out_main + out_sparse


# ============================================================================
# 递归替换工具函数
# ============================================================================

def _should_skip_module(fullname: str, skip_substrings) -> bool:
    """检查模块全名是否匹配任一排除子串。"""
    return any(substring in fullname for substring in (skip_substrings or ()))


def replace_hif4_layers_recursive(
    module: nn.Module,
    prefix: str = "",
    skip_substrings=None,
    qtype: str = DEFAULT_FP4_QTYPE,
) -> int:
    """递归将 nn.Linear 替换为 HiF4Linear（纯 HiF4 / NVFP4 伪量化路径）。"""
    qtype = resolve_fp4_qtype(qtype)
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            # 跳过 LM Head、Embedding 等
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue

            hif4_layer = HiF4Linear(child, qtype=qtype)
            setattr(module, name, hif4_layer)
            count += 1
        else:
            count += replace_hif4_layers_recursive(
                child, prefix=fullname, skip_substrings=skip_substrings, qtype=qtype
            )

    return count


def replace_morpho_hif4_layers_recursive(
    module: nn.Module,
    config,
    prefix: str = "",
) -> int:
    """递归将 nn.Linear 替换为 MorphoHiF4Linear（Morpho + HiF4/NVFP4 融合路径）。"""
    # 提前解析一次，让非法的 fp4_qtype 在替换开始前就报错
    resolve_fp4_qtype_from_config(config)
    skip_substrings = getattr(config.quant, "skip_module_substrings", None)
    count = 0
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if "lm_head" in name or "output" in name:
                continue
            if _should_skip_module(fullname, skip_substrings):
                continue

            morpho_layer = MorphoHiF4Linear(child, config, fullname)
            setattr(module, name, morpho_layer)
            count += 1
        else:
            count += replace_morpho_hif4_layers_recursive(
                child, config, prefix=fullname
            )

    return count
