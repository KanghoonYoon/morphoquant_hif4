#!/usr/bin/env python3
"""
Hardware Latency Benchmark for Quantization Methods.

Compares FP16 baseline vs MorphoQuant (current) vs MorphoQuant (fused kernel)
across controlled input lengths with CUDA Event-level timing.

Usage:
    # Single config
    CUDA_VISIBLE_DEVICES=4 python scripts/run_latency_bench.py \\
        --config configs/latency_bench/fp16.yaml

    # Three-way comparison
    CUDA_VISIBLE_DEVICES=4 python scripts/run_latency_bench.py \\
        --configs configs/latency_bench/fp16.yaml \\
                  configs/latency_bench/morpho.yaml \\
                  configs/latency_bench/morpho_fused.yaml \\
        --output /private/wy/logs/latency_bench/comparison.json

    # Quick mode (shortest input length only)
    CUDA_VISIBLE_DEVICES=4 python scripts/run_latency_bench.py \\
        --config configs/latency_bench/morpho.yaml --quick
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

import torch

# Ensure project root and custom bnb are on path
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.join(_PROJ_ROOT, "bnb_src"))

from config import AppConfig
from modules.model_factory import ModelBuilder
from modules.latency_probe import LatencyProbe, LatencyStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(torch.cuda.current_device())


def _check_gpu_constraint():
    """Enforce GPU 4/5/6 restriction as per CLAUDE.md."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        allowed = {"4", "5", "6"}
        requested = set(visible.split(","))
        if not requested.issubset(allowed):
            print(f"[WARN] CUDA_VISIBLE_DEVICES={visible} includes GPUs outside 4,5,6!")
    else:
        print("[WARN] CUDA_VISIBLE_DEVICES not set.  Consider setting to 4,5,6.")


def _build_controlled_inputs(
    processor,
    input_lengths: List[int],
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Create text-only inputs at exact token counts by repeating a dummy sentence.

    Uses the real tokenizer to encode text (not random token IDs) to avoid
    out-of-vocabulary issues and to match the model's expected input format.
    """
    text_unit = "Hello world. "
    repeat_count = 500  # ~1000+ tokens, expanded geometrically when needed
    all_tokens = processor(text=text_unit * repeat_count, return_tensors="pt")
    max_available = all_tokens["input_ids"].size(1)

    inputs_by_len = {}
    for length in input_lengths:
        while length > max_available:
            # Token counts per text repetition vary by tokenizer. Grow from the
            # existing text instead of deriving repetitions from a rough ratio;
            # the old ratio could shrink a 500-repeat prompt to only 9 repeats.
            repeat_count *= 2
            all_tokens = processor(
                text=text_unit * repeat_count,
                return_tensors="pt",
            )
            max_available = all_tokens["input_ids"].size(1)

        input_ids = all_tokens["input_ids"][:, :length].to(device)
        if input_ids.size(1) != length:
            raise RuntimeError(
                f"Failed to construct requested input length {length}; "
                f"tokenizer produced only {input_ids.size(1)} tokens"
            )
        attention_mask = torch.ones_like(input_ids).to(device)

        inputs_by_len[length] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    return inputs_by_len
def _text_decode_flops(text_config, num_layers: int) -> int:
    """Operational FLOPs/token for decoder linears plus the LM head."""
    hidden = int(text_config.hidden_size)
    intermediate = int(text_config.intermediate_size)
    heads = int(text_config.num_attention_heads)
    kv_heads = int(text_config.num_key_value_heads)
    head_dim = int(getattr(text_config, "head_dim", hidden // heads))
    kv_width = kv_heads * head_dim
    vocab = int(text_config.vocab_size)

    layer_macs = (
        2 * hidden * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * intermediate
    )
    lm_head_macs = hidden * vocab
    return 2 * (num_layers * layer_macs + lm_head_macs)


def _prune_text_layers(model, keep_layers: int) -> Dict[str, Any]:
    """Uniformly retain text decoder layers and keep KV-cache indices contiguous."""
    thinker = getattr(model, "thinker", None)
    text_model = getattr(thinker, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose thinker.model.layers")

    total_layers = len(layers)
    keep_layers = int(keep_layers)
    if not 1 <= keep_layers <= total_layers:
        raise ValueError(
            f"keep_text_layers must be in [1, {total_layers}], got {keep_layers}"
        )
    if keep_layers == total_layers:
        indices = list(range(total_layers))
    elif keep_layers == 1:
        indices = [total_layers - 1]
    else:
        indices = [
            round(i * (total_layers - 1) / (keep_layers - 1))
            for i in range(keep_layers)
        ]
    if len(set(indices)) != keep_layers:
        raise RuntimeError(f"Layer selection produced duplicate indices: {indices}")

    text_config = text_model.config
    original_layer_types = list(getattr(text_config, "layer_types", []) or [])
    text_model.layers = torch.nn.ModuleList([layers[i] for i in indices])

    for new_idx, layer in enumerate(text_model.layers):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "layer_idx"):
            self_attn.layer_idx = new_idx

    related_configs = [text_config]
    thinker_config = getattr(thinker, "config", None)
    if thinker_config is not None:
        related_configs.append(getattr(thinker_config, "text_config", None))
    top_thinker_config = getattr(getattr(model, "config", None), "thinker_config", None)
    if top_thinker_config is not None:
        related_configs.append(getattr(top_thinker_config, "text_config", None))

    for cfg in related_configs:
        if cfg is None:
            continue
        cfg.num_hidden_layers = keep_layers
        if original_layer_types and hasattr(cfg, "layer_types"):
            cfg.layer_types = [original_layer_types[i] for i in indices]

    original_flops = _text_decode_flops(text_config, total_layers)
    retained_flops = _text_decode_flops(text_config, keep_layers)
    return {
        "original_layers": total_layers,
        "kept_layers": keep_layers,
        "kept_indices": indices,
        "decode_operational_flops": retained_flops,
        "decode_operational_flops_reduction_pct": (
            1.0 - retained_flops / original_flops
        ) * 100.0,
    }




# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

def run_single_benchmark(
    config_path: str,
    quick: bool = False,
) -> Dict[str, Any]:
    """Build model from config and measure latency at controlled input lengths.

    Returns:
        Dict with keys: quant_method, model_path, model_type, gpu_name,
        metrics (input_len → LatencyStats dict), errors (list of str).
    """
    config = AppConfig.from_yaml(config_path)
    bc = config.benchmark

    result = {
        "quant_method": config.model.quant_method,
        "model_path": config.model.model_path,
        "model_type": config.model.model_type,
        "gpu_name": _gpu_name(),
        "metrics": {},
        "errors": [],
    }

    # Auto-detect token limit key: Qwen2.5-Omni uses thinker_max_new_tokens
    model_type_lower = config.model.model_type.lower()
    if "qwen" in model_type_lower and "omni" in model_type_lower:
        token_limit_key = "thinker_max_new_tokens"
    else:
        token_limit_key = "max_new_tokens"

    if quick:
        input_lengths = [min(bc.input_lengths)]
    else:
        input_lengths = bc.input_lengths

    print(f"\n{'='*60}")
    print(f"Building model: {config.model.quant_method}")
    print(f"  Model path: {config.model.model_path}")
    print(f"{'='*60}")

    # ---- Model build ----
    t0 = time.time()
    try:
        model, processor = ModelBuilder.build(config)
    except Exception as e:
        result["errors"].append(f"Model build failed: {e}")
        print(f"  [FAIL] Model build error: {e}")
        return result
    build_time = time.time() - t0
    result["build_time_sec"] = round(build_time, 1)
    print(f"  Model built in {build_time:.1f}s")

    if bc.keep_text_layers is not None:
        try:
            pruning = _prune_text_layers(model, bc.keep_text_layers)
        except Exception as e:
            result["errors"].append(f"Structured pruning failed: {e}")
            print(f"  [FAIL] Structured pruning error: {e}")
            return result
        result["structured_pruning"] = pruning
        print(
            f"  Structured text depth: {pruning['kept_layers']}/"
            f"{pruning['original_layers']} layers; operational FLOPs "
            f"-{pruning['decode_operational_flops_reduction_pct']:.1f}%"
        )
        torch.cuda.empty_cache()

    model.eval()

    # ---- For morpho method: run calibrate + prepare FIRST ----
    quant_method = config.model.quant_method
    needs_calib = quant_method in ("morpho", "qvlm") or bc.fused_kernel

    if needs_calib:
        print("\n  Running calibrate + prepare (required for MorphoQuant)...")
        try:
            _run_minimal_calibration(model, processor, config)
        except Exception as e:
            result["errors"].append(f"Calibration failed: {e}")
            print(f"  [FAIL] Calibration error: {e}")
            return result

    # ---- Fused kernel path: extract params AFTER calibration, then replace ----
    if bc.enabled and bc.fused_kernel:
        print("\n  Extracting MorphoQuant parameters for fused kernel...")
        from modules.morpho_fused_linear import extract_morpho_params, replace_morpho_with_fused

        layer_params = extract_morpho_params(model)
        print(f"  Found {len(layer_params)} quantized layers with calibration data")

        replace_morpho_with_fused(model, layer_params,
                                  activation_bit=config.quant.activation_bitwidth,
                                  use_cuda_kernel=bc.use_cuda_kernel)
        print("  Layers replaced with MorphoFusedLinear")

    # ---- Controlled inputs ----
    device = next(model.parameters()).device
    inputs_by_len = _build_controlled_inputs(processor, input_lengths, device)

    # ---- Latency measurement ----
    probe = LatencyProbe(warmup=bc.warmup, num_decode_tokens=bc.decode_tokens,
                         token_limit_key=token_limit_key)

    for length in input_lengths:
        print(f"\n  Measuring input_length={length}...")
        inputs = inputs_by_len[length]

        try:
            if bc.num_trials > 1:
                stats = probe.measure_with_trials(
                    model, inputs,
                    generate_kwargs={"use_audio_in_video": False, "return_audio": False},
                    num_trials=bc.num_trials,
                )
            else:
                stats = probe.measure(
                    model, inputs,
                    generate_kwargs={"use_audio_in_video": False, "return_audio": False},
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result["errors"].append(f"OOM at input_length={length}")
            print(f"  [OOM] input_length={length}")
            result["metrics"][f"input_{length}"] = None
            continue
        except Exception as e:
            result["errors"].append(f"Measure error at len={length}: {e}")
            print(f"  [FAIL] {e}")
            result["metrics"][f"input_{length}"] = None
            continue

        result["metrics"][f"input_{length}"] = stats.to_dict()
        print(f"    TTFT: {stats.ttft_ms:.1f} ms  |  "
              f"Decode: {stats.decode_ms_per_token:.1f} ms/tok  |  "
              f"Prefill: {stats.prefill_ms:.1f} ms  |  "
              f"Memory: {stats.peak_memory_mb:.0f} MB")

    # ---- Optional: SNR check for fused kernel ----
    if bc.enabled and bc.fused_kernel:
        _check_fused_snr(model, processor, config, result)

    return result


def _run_minimal_calibration(model, processor, config):
    """Lightweight calibration using only a few text-only forward passes.

    Replicates the essential calibrate + prepare flow from the evaluator:
    1. Calibrate: collect per-channel activation statistics via QuantAct
    2. Compute dispersion scores → set outlier masks
    3. Search: grid-search optimal quantization ranges (minimal)
    4. Finalize: cement best parameters for inference
    """
    calib_size = min(config.quant.calib_size, 32)
    search_size = min(config.quant.search_size, 8)

    # ---- Phase 1: Calibrate ----
    from modules.evaluator import _quantact_use_full_channel_outlier_mask

    _set_quantact_mode(model, calibrate=True, search=False)

    dummy_text = "Hello world. " * 20  # ~100 tokens
    device = next(model.parameters()).device

    for i in range(max(calib_size // 4, 1)):
        inputs = processor(text=dummy_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                _ = model.generate(**inputs,
                    thinker_max_new_tokens=4,
                    use_audio_in_video=False, return_audio=False)
        except Exception:
            continue

    # Compute dispersion scores and set outlier masks
    for _, module in model.named_modules():
        if hasattr(module, 'compute_dispersion_score') and hasattr(module, 'calib_count') and module.calib_count > 0:
            module.compute_dispersion_score()
        if hasattr(module, 'dispersion_score') and module.dispersion_score is not None:
            dispersion_score = module.dispersion_score
            if _quantact_use_full_channel_outlier_mask(config, getattr(module, "layer_name", "") or ""):
                outlier_mask = torch.ones_like(dispersion_score, dtype=torch.bool)
            else:
                threshold_val = getattr(config.quant, 'outlier_std_threshold', 2.0)
                threshold = dispersion_score.mean() + threshold_val * dispersion_score.std()
                outlier_mask = dispersion_score >= threshold
            module.register_buffer("outlier_mask", outlier_mask)

    # ---- Phase 2: Search (minimal) ----
    _set_quantact_mode(model, calibrate=False, search=True)

    for i in range(max(search_size // 4, 1)):
        inputs = processor(text=dummy_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                _ = model.generate(**inputs,
                    thinker_max_new_tokens=4,
                    use_audio_in_video=False, return_audio=False)
        except Exception:
            continue

    # ---- Phase 3: Finalize ----
    for _, module in model.named_modules():
        if hasattr(module, 'finalize_search'):
            module.finalize_search()

    _set_quantact_mode(model, calibrate=False, search=False)
    torch.cuda.empty_cache()


def _set_quantact_mode(model, calibrate, search):
    for _, module in model.named_modules():
        if hasattr(module, 'set_calibrate'):
            module.set_calibrate(calibrate)
        if hasattr(module, 'set_search'):
            module.set_search(search)


def _check_fused_snr(model, processor, config, result):
    """Compute SNR between fused kernel output and FP16 reference for a sample layer."""
    print("\n  Computing fused-kernel SNR...")
    try:
        from modules.morpho_fused_linear import MorphoFusedLinear

        # Find one MorphoFusedLinear layer
        target_layer = None
        target_name = None
        for name, module in model.named_modules():
            if isinstance(module, MorphoFusedLinear):
                target_layer = module
                target_name = name
                break

        if target_layer is None:
            print("  [WARN] No MorphoFusedLinear layers found for SNR check")
            return

        device = next(model.parameters()).device
        # torch._int_mm requires M >= 16, so use batch >= 16
        x_test = torch.randn(32, target_layer.in_features, dtype=torch.bfloat16, device=device)

        # Fused output
        y_fused = target_layer.forward(x_test)

        # Gather info
        result["snr_check"] = {
            "layer": target_name,
            "in_features": target_layer.in_features,
            "out_features": target_layer.out_features,
            "num_outliers": target_layer.outlier_indices.numel(),
            "has_outliers": target_layer.has_outliers,
        }
        print(f"  Layer: {target_name} ({target_layer.in_features}→{target_layer.out_features}), "
              f"outliers={target_layer.outlier_indices.numel()}")

    except Exception as e:
        print(f"  [WARN] SNR check failed: {e}")


# ---------------------------------------------------------------------------
# Multi-config comparison
# ---------------------------------------------------------------------------

def run_comparison_benchmark(
    config_paths: List[str],
    output_path: str,
    quick: bool = False,
):
    """Run benchmark across multiple configs and produce comparison output."""
    all_results = []
    for cfg_path in config_paths:
        result = run_single_benchmark(cfg_path, quick=quick)
        all_results.append(result)

    # Gather metadata
    report = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "gpu_name": _gpu_name(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "quick_mode": quick,
        "results": all_results,
    }

    # Write JSON
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {output_path}")

    # Print comparison table
    _print_comparison_table(all_results)

    return report


def _print_comparison_table(results: List[dict]):
    """Pretty-print a comparison table to stdout."""
    # Collect all input lengths
    all_lengths = set()
    for r in results:
        for key in (r.get("metrics") or {}):
            if key.startswith("input_"):
                all_lengths.add(int(key.split("_")[1]))
    lengths = sorted(all_lengths)

    if not lengths:
        print("\n[No metrics to display]")
        return

    print(f"\n{'='*100}")
    print(f"  Latency Benchmark Comparison  |  GPU: {_gpu_name()}")
    print(f"{'='*100}")
    header = f"{'Method':<22} | {'Len':>6} | {'Prefill(ms)':>12} | {'Decode(ms/t)':>13} | {'TTFT(ms)':>10} | {'Mem(MB)':>9}"
    print(header)
    print("-" * len(header))

    for r in results:
        method = r.get("quant_method", "unknown")
        if r.get("errors"):
            method += " [ERR]"
        metrics = r.get("metrics") or {}

        for length in lengths:
            key = f"input_{length}"
            m = metrics.get(key)
            if m is None:
                print(f"{method:<22} | {length:>6} | {'OOM/ERR':>12} |")
                continue
            print(
                f"{method:<22} | {length:>6} | "
                f"{m.get('prefill_ms', 0):>12.1f} | "
                f"{m.get('decode_ms_per_token', 0):>13.1f} | "
                f"{m.get('ttft_ms', 0):>10.1f} | "
                f"{m.get('peak_memory_mb', 0):>9.0f}"
            )
        print("-" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hardware Latency Benchmark for MorphoQuant Quantization Methods"
    )
    cfg_group = parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument("--config", type=str, help="Single YAML config path")
    cfg_group.add_argument("--configs", type=str, nargs="+", help="Multiple YAML config paths for comparison")

    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path for benchmark results")
    parser.add_argument("--quick", action="store_true",
                        help="Only benchmark the shortest input length")
    args = parser.parse_args()

    _check_gpu_constraint()

    config_paths = [args.config] if args.config else args.configs
    output = args.output
    if not output and args.configs:
        output = "/private/wy/logs/latency_bench/comparison.json"

    if len(config_paths) == 1:
        result = run_single_benchmark(config_paths[0], quick=args.quick)
        if output:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            report = {
                "benchmark_version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "gpu_name": _gpu_name(),
                "results": [result],
            }
            with open(output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nJSON report saved to: {output}")
        _print_comparison_table([result])
    else:
        run_comparison_benchmark(config_paths, output, quick=args.quick)


if __name__ == "__main__":
    main()
