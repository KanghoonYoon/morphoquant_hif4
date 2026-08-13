import os
import random
import argparse

from config import AppConfig
from modules.model_factory import ModelBuilder
from modules.evaluator import MMMUEvaluator

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
BASE_DIR = "/private/wy"


parser = argparse.ArgumentParser(description="MMMU Quantization Inference")
parser.add_argument("--config", type=str, default="configs/qwen2.5-omni-3b/morpho/mmmu_morpho.yaml", help="Path to the yaml configuration file")
args = parser.parse_args()

config = AppConfig.from_yaml(args.config)
random.seed(config.run.seed)

# Keep the same behavior as the old MMMU script: HiF8/HiF4 simulation forces BF16 loading.
QUANT_METHOD = config.model.quant_method
SIMULATE_HIF8 = config.quant.simulate_hif8
SIMULATE_HIF4 = config.quant.simulate_hif4
APPLY_HOOKS = config.model.apply_hooks
SAVE_ATTENTION_MAPS = config.model.save_attention_maps
if SIMULATE_HIF8:
    if QUANT_METHOD != 'morpho_withhif8':
        print(">>> HiF8 Simulation ENABLED: Forcing quant_method='bf16' (except for morpho_withhif8).")
        config.model.quant_method = "bf16"
        QUANT_METHOD = "bf16"
    else:
        print(">>> Morpho + HiF8 Beta Path ENABLED.")
    
    import sys
    sys.path.append("/private/wy/MorphoQuant/modules/HiF8_NPU_GPU_simulator/GPU_Cuda/hif8_cuda")
    try:
        from quant_cy import quant_dequant_float, QType
    except ImportError:
        pass

if SIMULATE_HIF4:
    if QUANT_METHOD != 'morpho_withhif4':
        print(">>> HiF4 Simulation ENABLED: Forcing quant_method='bf16' (except for morpho_withhif4).")
        config.model.quant_method = "bf16"
        QUANT_METHOD = "bf16"
    else:
        print(">>> Morpho + HiF4 Beta Path ENABLED.")
    
    import sys
    sys.path.insert(0, "/private/wy/MorphoQuant/modules/HiFloat4-main/hif4_gpu")
    try:
        from quant_cy import quant_dequant_float, QType
    except ImportError:
        pass

if QUANT_METHOD in ("morpho", "morpho_withhif8", "morpho_withhif4"):
    from bitsandbytes.quantization_utils.quant_modules import QuantAct

    # Enable debug mode for QuantAct
    debug_quant_act = QuantAct(activation_bit=4)
    debug_quant_act.set_debug(True)
    print("QuantAct debug mode enabled.")

print("=" * 50)
print(f"  Model Path:           {config.model.model_path}")
print(f"  Model Type:           {getattr(config.model, 'model_type', 'qwen2_5_omni')}")
print(f"  Num Samples/Subj:     {config.data.num_samples if config.data.num_samples != -1 else None}")
print(f"  Quant Method:         {config.model.quant_method}")
print(f"  Simulate HiF8:        {config.quant.simulate_hif8}")
print(f"  Simulate HiF4:        {config.quant.simulate_hif4}")
print(f"  Simulate SmoothQuant: {config.quant.simulate_smoothquant}")
print(f"  Target Subject:       {config.data.target_subject}")
print(f"  Dataset Dir:          {config.data.dataset_dir}")
print(f"  Run PCA:              {config.data.run_pca}")
print(f"  Apply Hooks:          {APPLY_HOOKS}")
print(f"  Save Attn Maps:       {SAVE_ATTENTION_MAPS}")
print("=" * 50)

from visualizer import (
    register_activation_hooks,
    plot_activation_histograms,
    register_attention_hooks,
    plot_and_save_attention_maps
)
from datetime import datetime

def generate_filename_with_time(num_samples):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_str = f"{num_samples}samples" if num_samples is not None else "allsamples"
    filename = f"{num_str}_{timestamp}.csv"
    return filename

model, processor = ModelBuilder.build(config)

if APPLY_HOOKS:
    v_len = None
    hooks = register_activation_hooks(model, v_len)
    print(f"✅ 已注册 {len(hooks)} 个 hook，用于捕获激活分布。")

if SAVE_ATTENTION_MAPS:
    model.config.output_attentions = True
    if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "config"):
        model.thinker.model.config.output_attentions = True
        
    attn_hooks = register_attention_hooks(model)
    print(f"✅ 已注册 {len(attn_hooks)} 个 hook，用于捕获 Attention Map。")

evaluator = MMMUEvaluator(model, processor, config, BASE_DIR)

evaluator.calibrate()
evaluator.prepare()

filename = generate_filename_with_time(config.data.num_samples)
out_log = os.path.join(config.run.save_dir, filename)
print(f"开始在 MMMU 测试集上评测模型...")
evaluator.evaluate(save_dir=config.run.save_dir)

if APPLY_HOOKS:
    plot_activation_histograms("/private/wy/logs/MorphoQuant/activation_hist")
