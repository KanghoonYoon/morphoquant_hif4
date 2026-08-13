#!/usr/bin/env python3
"""
VRAM memory usage test for MorphoQuant quantization configurations.

Loads a model from a YAML config, runs calibration → prepare → a few inference
samples on the ScienceQA dataset, and records peak VRAM at each stage.

Usage:
    python scripts/test_vram.py --config configs/qwen2.5-omni-3b/qlora/scienceqa_qlora_w4a16.yaml
    python scripts/test_vram.py --config <path> --num_samples 5
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

import torch
import torch.nn as nn
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_DIR = "/private/wy"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from config import AppConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GB = 1024**3


def _cuda_mem(device=None):
    """Return (allocated_gb, reserved_gb, max_allocated_gb)."""
    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
    ma = torch.cuda.max_memory_allocated(device) / GB
    a = torch.cuda.memory_allocated(device) / GB
    r = torch.cuda.memory_reserved(device) / GB
    return a, r, ma


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def free_gpu():
    """Aggressively free GPU memory including driver-level cached allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test_vram(config_path: str, num_samples: int = 5) -> dict:
    """Load model, run pipeline, return VRAM stats dict."""
    result = {
        "config_path": config_path,
        "status": "ok",
        "model": "",
        "method": "",
        "precision": "",
        "model_load_vram_gb": 0.0,
        "calibrate_peak_vram_gb": 0.0,
        "prepare_peak_vram_gb": 0.0,
        "inference_peak_vram_gb": 0.0,
        "total_peak_vram_gb": 0.0,
        "error": "",
    }

    # ------------------------------------------------------------------
    # 0. Load config
    # ------------------------------------------------------------------
    free_gpu()
    print(f"\n{'='*60}")
    print(f"Loading config: {config_path}")
    print(f"{'='*60}")

    config = AppConfig.from_yaml(config_path)
    result["model"] = config.model.model_path.rstrip("/").split("/")[-1]
    result["method"] = config.model.quant_method

    # Derive precision label from config
    quant = config.quant
    if quant.simulate_smoothquant:
        wb = quant.smoothquant_weight_bits
        ab = quant.smoothquant_act_bits or 16
        result["precision"] = f"W{wb}A{ab}"
    elif quant.simulate_awq:
        result["precision"] = f"W{quant.awq_weight_bits}A16"
    elif quant.simulate_mbq:
        wb = quant.mbq_weight_bits
        ab = quant.mbq_act_bits or 16
        result["precision"] = f"W{wb}A{ab}"
    elif quant.simulate_mquant:
        result["precision"] = f"W{quant.mquant_weight_bits}A{quant.mquant_act_bits}"
    elif quant.simulate_freeact:
        result["precision"] = f"W{quant.freeact_weight_bits}A{quant.freeact_act_bits}"
    elif config.model.quant_method in ("morpho_withhif8",):
        result["precision"] = "W8A8"
        result["method"] = "Morpho+HiF8"
    elif config.model.quant_method in ("morpho_withhif4",):
        result["precision"] = "W4A4"
        result["method"] = "Morpho+HiF4"
    elif config.model.quant_method in ("morpho", "qvlm"):
        ab = quant.activation_bitwidth
        result["precision"] = f"W4A{ab}"
    elif config.model.quant_method == "qlora":
        result["precision"] = "W4A16"
    elif config.model.quant_method == "quarot":
        result["precision"] = f"W{quant.quarot_weight_bits}A{quant.quarot_act_bits}"
    else:
        result["precision"] = f"W{quant.activation_bitwidth}A{quant.activation_bitwidth}"

    # Override num_samples for quick VRAM-only test; keep calibration/search as-is
    config.data.num_samples = num_samples

    print(f"  Model:     {result['model']}")
    print(f"  Method:    {result['method']}")
    print(f"  Precision: {result['precision']}")
    print(f"  Quant method (raw): {config.model.quant_method}")
    print(f"  Num samples for inference: {num_samples}")

    # Setup HiF import paths (mirrors wy_inference_scienceqa.py)
    quant_method = config.model.quant_method
    if config.quant.simulate_hif8 and quant_method != "morpho_withhif8":
        quant_method = "bf16"
        sys.path.append(os.path.join(PROJECT_DIR, "modules", "HiF8_NPU_GPU_simulator", "GPU_Cuda", "hif8_cuda"))
    elif config.quant.simulate_hif4 and quant_method not in ("morpho_withhif4",):
        quant_method = "bf16"
        sys.path.insert(0, os.path.join(PROJECT_DIR, "modules", "HiFloat4-main", "hif4_gpu"))

    # ------------------------------------------------------------------
    # 1. Build model — track peak VRAM
    # ------------------------------------------------------------------
    print("\n[1/4] Building model...")
    free_gpu()
    reset_peak()

    from modules.model_factory import ModelBuilder

    try:
        model, processor = ModelBuilder.build(config)
    except Exception as e:
        result["status"] = "model_build_error"
        result["error"] = str(e)
        traceback.print_exc()
        return result

    # Set morpho search hyperparams (mirrors wy_inference_scienceqa.py)
    if config.model.quant_method in ("morpho", "qvlm", "morpho_withhif8", "morpho_withhif4"):
        from bitsandbytes.quantization_utils.quant_modules import QuantAct
        lp_norm = getattr(config.quant, "lp_norm", 2.0)
        use_cosine_loss = getattr(config.quant, "use_cosine_loss", False)
        for name, module in model.named_modules():
            if isinstance(module, QuantAct):
                module.set_lp_norm(lp_norm)
                module.set_cosine_loss(use_cosine_loss)

    torch.cuda.synchronize()
    model_load_vram = torch.cuda.max_memory_allocated() / GB
    result["model_load_vram_gb"] = round(model_load_vram, 2)
    print(f"  Model load peak VRAM: {model_load_vram:.2f} GB")

    # ------------------------------------------------------------------
    # 2. Calibrate
    # ------------------------------------------------------------------
    print("\n[2/4] Calibrating...")
    reset_peak()

    from modules.evaluator import ScienceQAEvaluator

    evaluator = ScienceQAEvaluator(model, processor, config, BASE_DIR)

    try:
        evaluator.calibrate()
    except Exception as e:
        result["status"] = "calibrate_error"
        result["error"] = str(e)
        traceback.print_exc()
        free_gpu()
        return result

    torch.cuda.synchronize()
    calibrate_vram = torch.cuda.max_memory_allocated() / GB
    result["calibrate_peak_vram_gb"] = round(calibrate_vram, 2)
    print(f"  Calibrate peak VRAM: {calibrate_vram:.2f} GB")

    # ------------------------------------------------------------------
    # 3. Prepare (search for morpho methods)
    # ------------------------------------------------------------------
    print("\n[3/4] Preparing (search)...")
    reset_peak()

    try:
        evaluator.prepare()
    except Exception as e:
        result["status"] = "prepare_error"
        result["error"] = str(e)
        traceback.print_exc()
        free_gpu()
        return result

    torch.cuda.synchronize()
    prepare_vram = torch.cuda.max_memory_allocated() / GB
    result["prepare_peak_vram_gb"] = round(prepare_vram, 2)
    print(f"  Prepare peak VRAM: {prepare_vram:.2f} GB")

    # ------------------------------------------------------------------
    # 4. Inference — run a few samples, track peak VRAM
    # ------------------------------------------------------------------
    print(f"\n[4/4] Running {num_samples} inference samples...")
    from modules.evaluator import ScienceQADataset, _normalize_conversation_for_template
    from qwen_omni_utils import process_mm_info

    # Ensure QuantAct is in eval mode
    if config.model.quant_method in ("morpho", "qvlm", "morpho_withhif8", "morpho_withhif4"):
        from bitsandbytes.quantization_utils.quant_modules import QuantAct
        for _, module in model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=False)
                module.set_search(search=False)

    dataset = ScienceQADataset(
        base_dir=BASE_DIR,
        problems_path=evaluator.problems_path,
        split_path=evaluator.split_path,
        split="test",
        only_samples_with_images=config.data.only_samples_with_images,
        max_samples=num_samples,
    )

    reset_peak()
    is_internvl = evaluator._is_internvl()
    inference_samples = 0

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Inference"):
        data = dataset[i]
        try:
            with torch.no_grad():
                if is_internvl:
                    evaluator._inference_internvl(data, max_new_tokens=10)
                else:
                    conversation = data["conversation"]
                    # Mirror _inference() logic
                    conv_for_template = _normalize_conversation_for_template(conversation, processor)
                    text = processor.apply_chat_template(
                        conv_for_template, add_generation_prompt=True, tokenize=False
                    )
                    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                    inputs = processor(
                        text=text, audio=audios, images=images, videos=videos,
                        return_tensors="pt", padding=True, use_audio_in_video=False,
                    )
                    inputs = inputs.to(model.device).to(model.dtype)
                    evaluator._set_mquant_ctx(conversation, inputs)
                    model.generate(
                        **inputs,
                        thinker_max_new_tokens=10,
                        use_audio_in_video=False,
                        return_audio=False,
                    )
            inference_samples += 1
        except torch.cuda.OutOfMemoryError:
            print(f"  [WARN] CUDA OOM on sample {i}, skipping remaining samples")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  [WARN] Sample {i} error: {e}")
            continue

    torch.cuda.synchronize()
    inference_peak = torch.cuda.max_memory_allocated() / GB
    result["inference_peak_vram_gb"] = round(inference_peak, 2)
    result["num_inference_samples"] = inference_samples
    print(f"  Inference peak VRAM: {inference_peak:.2f} GB ({inference_samples} samples)")

    # Total peak = max of all phases
    result["total_peak_vram_gb"] = round(
        max(model_load_vram, calibrate_vram, prepare_vram, inference_peak), 2
    )
    print(f"  Total peak VRAM:    {result['total_peak_vram_gb']:.2f} GB")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    del model, processor, evaluator, dataset
    free_gpu()

    print(f"\nResult: {json.dumps(result, ensure_ascii=False)}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRAM memory test for quantization configs")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of inference samples (default: 5)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")
    print(f"GPU (CUDA_VISIBLE_DEVICES): {gpu_id}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Initial VRAM: {torch.cuda.memory_allocated()/GB:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved()/GB:.2f} GB reserved")

    result = test_vram(args.config, num_samples=args.num_samples)

    # Print final JSON line for easy parsing
    print(f"\n__RESULT_JSON__ {json.dumps(result, ensure_ascii=False)}")

    if result["status"] != "ok":
        sys.exit(1)
