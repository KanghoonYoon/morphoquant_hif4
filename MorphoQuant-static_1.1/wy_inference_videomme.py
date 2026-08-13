import os
import sys

import json
import re
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import argparse

from tqdm import tqdm
from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoConfig, AwqConfig
from accelerate import init_empty_weights
from datetime import datetime
from qwen_omni_utils import process_mm_info
from collections import defaultdict
from config import AppConfig

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

BASE_DIR = "/private/wy"

parser = argparse.ArgumentParser(description="Qwen2.5-Omni Quantization Inference")
parser.add_argument("--config", type=str, default="configs/qwen2.5-omni-3b/morpho/videomme_morpho.yaml", help="Path to the yaml configuration file")
args = parser.parse_args()

config = AppConfig.from_yaml(args.config)
random.seed(config.run.seed)

CALIB_SIZE = config.quant.calib_size
BATCH_SIZE = config.quant.batch_size
MODEL_PATH = config.model.model_path
SAVE_DIR = config.run.save_dir
NUM_SAMPLES = config.data.num_samples if config.data.num_samples != -1 else None
APPLY_HOOKS = config.model.apply_hooks
QUANT_METHOD = config.model.quant_method
DEBUG_QUANT_ACT = config.quant.debug_quant_act

GAMMA_INF = config.quant.gamma_inf
GAMMA_COS = config.quant.gamma_cos
SAVE_ATTENTION_MAPS = config.model.save_attention_maps
SAVE_PER_HEAD_ATTENTION = config.model.save_per_head_attention

KEEP_SPECIAL_LAYER = config.quant.keep_special_layer
ONLY_SAMPLES_WITH_IMAGES = config.data.only_samples_with_images
SIMULATE_HIF8 = config.quant.simulate_hif8
SIMULATE_HIF4 = config.quant.simulate_hif4
ACTIVATION_BITWIDTH = config.quant.activation_bitwidth
if SIMULATE_HIF8:
    if QUANT_METHOD != 'morpho_withhif8':
        print(">>> HiF8 Simulation ENABLED: Forcing quant_method='bf16' (except for morpho_withhif8).")
        QUANT_METHOD = 'bf16'
    else:
        print(">>> Morpho + HiF8 Beta Path ENABLED.")
    
    sys.path.append("/private/wy/MorphoQuant/modules/HiF8_NPU_GPU_simulator/GPU_Cuda/hif8_cuda")
    try:
        from quant_cy import quant_dequant_float, QType
    except ImportError:
        pass

if SIMULATE_HIF4:
    if QUANT_METHOD != 'morpho_withhif4':
        print(">>> HiF4 Simulation ENABLED: Forcing quant_method='bf16' (except for morpho_withhif4).")
        QUANT_METHOD = 'bf16'
    else:
        print(">>> Morpho + HiF4 Beta Path ENABLED.")

    # 后端 (CUDA quant_cy / NPU quant_cy_npu) 与 sys.path 由 hif4_layers 按设备解析
    from modules.hif4_layers import load_quant_cy
    try:
        load_quant_cy()
    except Exception as e:
        print(f">>> WARNING: FP4 simulator backend not loaded yet ({e}).")

if QUANT_METHOD in ("morpho", "morpho_withhif8", "morpho_withhif4"):
    from bitsandbytes.quantization_utils.quant_modules import QuantAct


from modules.device_utils import get_device_type

print("="*50)
print(f"  Device:               {get_device_type()}")
print(f"  Model Path:           {MODEL_PATH}")
print(f"  Calib Size:           {CALIB_SIZE}")
print(f"  Num Samples:          {NUM_SAMPLES}")
print(f"  Quant Method:         {QUANT_METHOD}")
print(f"  Keep Special Layer:   {KEEP_SPECIAL_LAYER}")
print(f"  Save Attn Maps:       {SAVE_ATTENTION_MAPS}")
print(f"  Only w/ Images:       {ONLY_SAMPLES_WITH_IMAGES}")
print(f"  Simulate HiF8:        {SIMULATE_HIF8}")
print(f"  Simulate HiF4:        {SIMULATE_HIF4}")
if SIMULATE_HIF4:
    print(f"  FP4 dtype:            {config.quant.fp4_qtype}")
print(f"  Simulate SmoothQuant: {config.quant.simulate_smoothquant}")
print("="*50)

from visualizer import (
    register_activation_hooks,
    plot_activation_histograms,
    register_attention_hooks,
    plot_and_save_attention_maps
)
def generate_filename_with_time(num_samples):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_str = f"{num_samples}samples" if num_samples is not None else "allsamples"
    filename = f"{num_str}_{timestamp}.csv"
    return filename

# ========== hook 相关代码结束 ==========

from modules.model_factory import ModelBuilder
model, processor = ModelBuilder.build(config)

if APPLY_HOOKS:
    # v_len = get_visual_token_length(model, processor)
    v_len = None
    # print(v_len)
    # sys.exit(0)

    # hooks = register_activation_hooks(model)
    hooks = register_activation_hooks(model, v_len)
    print(f"✅ 已注册 {len(hooks)} 个 hook，用于捕获激活分布。")

if SAVE_ATTENTION_MAPS:
    # 强制开启 output_attentions，确保 forward 返回 weights
    model.config.output_attentions = True
    # 针对 Qwen Omni 结构，可能需要深入到 thinker.model 设置
    if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "config"):
        model.thinker.model.config.output_attentions = True
        
    attn_hooks = register_attention_hooks(model)
    print(f"✅ 已注册 {len(attn_hooks)} 个 hook，用于捕获 Attention Map。")

# ========== 推理函数 ==========
if __name__ == "__main__":
    from modules.evaluator import VideoMMEEvaluator
    
    evaluator = VideoMMEEvaluator(model, processor, config, BASE_DIR)
    
    # 1. 批量校准 (仅 morpho 模式执行)
    evaluator.calibrate()
    
    # 2. 推理前搜索和准备
    evaluator.prepare()

    # 3. 评测
    filename = generate_filename_with_time(NUM_SAMPLES)
    out_log = os.path.join(SAVE_DIR, filename)

    print(f"开始在 Video-MME 测试集上评测模型，结果保存到 {out_log} ...")
    acc = evaluator.evaluate(save_path=out_log)
    
    # 4. 可视化探针图表
    if APPLY_HOOKS:
        plot_activation_histograms("/private/wy/logs/MorphoQuant/activation_hist")
