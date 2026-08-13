"""设备抽象层：统一 NVIDIA CUDA 与 Ascend NPU 的差异。

用法::

    from modules.device_utils import get_device, device_module, on_accelerator

    model.to(get_device())          # "npu" 或 "cuda"
    device_module().empty_cache()   # torch.npu / torch.cuda

设备类型在首次调用时探测一次并缓存。可用环境变量 ``MORPHOQUANT_DEVICE``
强制指定 (npu / cuda / cpu)，便于在同时具备两种加速器的机器上做 A/B 对比。
"""

import os

import torch

_VALID_DEVICES = ("npu", "cuda", "cpu")
_DEVICE_TYPE = None


def _detect_device_type() -> str:
    forced = os.environ.get("MORPHOQUANT_DEVICE", "").strip().lower()
    if forced:
        if forced not in _VALID_DEVICES:
            raise ValueError(
                f"MORPHOQUANT_DEVICE={forced!r} is invalid; expected one of {_VALID_DEVICES}."
            )
        return forced

    # torch_npu 必须先 import，torch.npu 才会被注册
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_type() -> str:
    """返回 'npu' / 'cuda' / 'cpu'（结果被缓存）。"""
    global _DEVICE_TYPE
    if _DEVICE_TYPE is None:
        _DEVICE_TYPE = _detect_device_type()
    return _DEVICE_TYPE


def get_device(index=None) -> str:
    """给 ``model.to(...)`` / ``device_map=`` 用的设备字符串。"""
    dev = get_device_type()
    return dev if index is None else f"{dev}:{index}"


def is_npu() -> bool:
    return get_device_type() == "npu"


def is_available() -> bool:
    """是否存在可用加速器（CPU 视为不可用）。"""
    return get_device_type() != "cpu"


def device_module():
    """返回 torch.npu / torch.cuda / torch.cpu 之一。"""
    dev = get_device_type()
    mod = getattr(torch, dev, None)
    if mod is None:
        raise RuntimeError(f"torch.{dev} is unavailable in this PyTorch build.")
    return mod


def on_accelerator(tensor) -> bool:
    """替代 ``tensor.is_cuda`` 的设备无关写法。

    没有加速器时一律返回 False，以保持"尚未搬到设备上就跳过量化"的惰性初始化语义。
    """
    dev = get_device_type()
    if dev == "cpu":
        return False
    return tensor.device.type == dev


def empty_cache():
    mod = device_module()
    if hasattr(mod, "empty_cache"):
        mod.empty_cache()


def synchronize():
    mod = device_module()
    if hasattr(mod, "synchronize"):
        mod.synchronize()


def memory_allocated() -> int:
    mod = device_module()
    return mod.memory_allocated() if hasattr(mod, "memory_allocated") else 0


def memory_reserved() -> int:
    mod = device_module()
    return mod.memory_reserved() if hasattr(mod, "memory_reserved") else 0


def oom_errors() -> tuple:
    """可用于 ``except`` 的 OOM 异常类型元组。

    CUDA 有专用的 ``torch.cuda.OutOfMemoryError``；NPU 目前没有对应类型，
    退化为 ``RuntimeError``——因此捕获后必须再用 :func:`is_oom` 确认，
    以免吞掉无关的 RuntimeError。
    """
    exc = getattr(device_module(), "OutOfMemoryError", None)
    return (exc,) if isinstance(exc, type) else (RuntimeError,)


def is_oom(exc: BaseException) -> bool:
    """确认捕获到的异常确实是显存/内存耗尽。"""
    named = getattr(device_module(), "OutOfMemoryError", None)
    if isinstance(named, type) and isinstance(exc, named):
        return True
    return "out of memory" in str(exc).lower()
