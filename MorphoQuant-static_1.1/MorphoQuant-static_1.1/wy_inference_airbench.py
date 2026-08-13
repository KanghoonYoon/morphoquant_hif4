import os
import sys
import random
import argparse

from config import AppConfig
from modules.model_factory import ModelBuilder
from modules.evaluator import AirBenchEvaluator

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_DIR = "/private/wy"

parser = argparse.ArgumentParser(description="Qwen2.5-Omni Quantization Inference on AIR-Bench")
parser.add_argument("--config", type=str, default="configs/qwen2.5-omni-3b/morpho/airbench_morpho.yaml", help="Path to the yaml configuration file")
args = parser.parse_args()

config = AppConfig.from_yaml(args.config)
random.seed(config.run.seed)

config.quant.skip_module_substrings = ["visual.blocks.", "thinker.visual.merger.mlp."]

meta_path = os.path.join(config.data.data_root, "Foundation_meta.json")
if not os.path.exists(meta_path):
    raise FileNotFoundError(
        f"Meta file not found at {meta_path}. "
        "Please set data.data_root in the config to the AIR-Bench Foundation root directory."
    )

if config.quant.simulate_hif8:
    if config.model.quant_method != 'morpho_withhif8':
        print(">>> HiF8 Simulation ENABLED: Forcing quant_method='bf16' (except for morpho_withhif8).")
        config.model.quant_method = "bf16"
    else:
        print(">>> Morpho + HiF8 Beta Path ENABLED.")
        
    sys.path.append("/private/wy/MorphoQuant/modules/HiF8_NPU_GPU_simulator/GPU_Cuda/hif8_cuda")
    try:
        from quant_cy import quant_dequant_float, QType  # noqa: F401
    except ImportError:
        pass

if config.quant.simulate_hif4:
    if config.model.quant_method != 'morpho_withhif4':
        print(">>> HiF4 Simulation ENABLED: Forcing quant_method='bf16' (except for morpho_withhif4).")
        config.model.quant_method = "bf16"
    else:
        print(">>> Morpho + HiF4 Beta Path ENABLED.")
        
    sys.path.insert(0, "/private/wy/MorphoQuant/modules/HiFloat4-main/hif4_gpu")
    try:
        from quant_cy import quant_dequant_float, QType  # noqa: F401
    except ImportError:
        pass

if config.model.quant_method in ("morpho", "morpho_withhif8", "morpho_withhif4"):
    from bitsandbytes.quantization_utils.quant_modules import QuantAct

print("=" * 50)
print(f"  Model Path:           {config.model.model_path}")
print(f"  Data Root:            {config.data.data_root}")
print(f"  Output File:          {config.data.output_file}")
print(f"  Num Samples:          {config.data.num_samples}")
print(f"  Quant Method:         {config.model.quant_method}")
print(f"  Keep Special Layer:   {config.quant.keep_special_layer}")
print(f"  Simulate HiF8:        {config.quant.simulate_hif8}")
print(f"  Simulate HiF4:        {config.quant.simulate_hif4}")
print(f"  Simulate SmoothQuant: {config.quant.simulate_smoothquant}")
print(f"  Run PCA:              {config.data.run_pca}")
print("=" * 50)

model, processor = ModelBuilder.build(config)

evaluator = AirBenchEvaluator(model, processor, config, BASE_DIR)

from visualizer import (
    register_activation_hooks,
    plot_activation_histograms,
    register_attention_hooks,
    plot_and_save_attention_maps
)

if config.model.apply_hooks:
    v_len = None
    hooks = register_activation_hooks(model, v_len)
    print(f"✅ 已注册 {len(hooks)} 个 hook，用于捕获激活分布。")

if config.model.save_attention_maps:
    model.config.output_attentions = True
    if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "config"):
        model.thinker.model.config.output_attentions = True
        
    attn_hooks = register_attention_hooks(model)
    print(f"✅ 已注册 {len(attn_hooks)} 个 hook，用于捕获 Attention Map。")

# Align with ScienceQA workflow: calibrate -> prepare -> evaluate.
evaluator.calibrate()
evaluator.prepare()
evaluator.evaluate(save_path=config.data.output_file)

if config.model.apply_hooks:
    plot_activation_histograms("/private/wy/logs/MorphoQuant/activation_hist_airbench")
