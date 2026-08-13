# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MorphoQuant applies **Morpho Activation Quantization** to multimodal models (Qwen2.5-Omni, InternVL2.5). It supports multiple quantization methods (BF16, INT8, AWQ, HQQ, SmoothQuant, QuaRot, MBQ, MQuant, QLoRA, and Morpho with HiF8/HiF4 simulation) via a unified YAML-driven pipeline.

The project depends on a **locally modified bitsandbytes** at `bnb_src/` — the standard pip bitsandbytes will NOT work for Morpho-related quantization.

## GPU Restriction

**只允许使用 4、5、6 号 GPU。** 禁止占用或修改 0-3 号显卡上的进程。所有推理/校准脚本必须通过 `CUDA_VISIBLE_DEVICES` 限制在 4、5、6 号 GPU 范围内。使用 `wy_inference.sh` 时通过 `DEFAULT_GPU` 环境变量或 `--gpu` 参数指定。

## Key Commands

```bash
# Run individual benchmarks
python wy_inference_mmmu.py --config configs/qwen2.5-omni-7b/morpho/mmmu_morpho.yaml
python wy_inference_scienceqa.py --config configs/qwen2.5-omni-7b/morpho/scienceqa_morpho.yaml
python wy_inference_videomme.py --config configs/qwen2.5-omni-7b/morpho/videomme_morpho.yaml
python wy_inference_airbench.py --config configs/qwen2.5-omni-7b/morpho/airbench_morpho.yaml

# Run via unified shell launcher (batch)
./wy_inference.sh mmmu                    # single benchmark
./wy_inference.sh all                     # all four benchmarks
./wy_inference.sh --gpu 0 scienceqa       # specific GPU
./wy_inference.sh all --dry-run           # preview commands only

# Smoke test (no heavy model loading)
python tests/smoke_morpho_internvl.py

# Override HuggingFace endpoint (default: hf-mirror.com)
HF_ENDPOINT=https://huggingface.co python wy_inference_mmmu.py --config ...
```

## Architecture

The pipeline follows a three-phase pattern for every benchmark:

```
AppConfig.from_yaml() → ModelBuilder.build() → Evaluator.calibrate() → Evaluator.prepare() → Evaluator.evaluate()
```

### 1. Configuration (`config.py`)
`AppConfig` is a dataclass with four groups: `ModelConfig`, `QuantConfig`, `DataConfig`, `RunConfig`. Loaded from YAML via `AppConfig.from_yaml(path)`.

### 2. Model Building (`modules/model_factory.py`)
`ModelBuilder.build(config)` returns `(model, processor)`. The builder:
- Routes by `config.model.model_type` (currently `qwen2_5_omni` or `internvl2_5`)
- Selects the build method by `config.model.quant_method`
- For `morpho`/`qvlm`: loads the model in 4-bit via bitsandbytes, then attaches `QuantAct` activation quantization modules via `_attach_morpho_quant_activation_to_bnb_linears()`
- For `morpho_withhif8`/`morpho_withhif4`: loads BF16 then replaces Linear layers with fused `MorphoHiF8Linear`/`MorphoHiF4Linear` wrappers
- For InternVL2.5: applies compatibility monkey-patches so `GenerationMixin` works with InternLM2

**Key design rule**: `_wrap_morpho_quantact_recursive()` wraps EVERY `nn.Linear` with `MorphoQuantActWrapper` EXCEPT layers whose local name is in `_SKIP_LOCAL_NAMES` (`lm_head`, `output`, `embed_tokens`, `patch_embed`, `patch_proj`, `pos_embed`, `cls_token`, `class_token`). This ensures attention projections (q_proj, k_proj, v_proj, o_proj) ARE wrapped.

### 3. Evaluation (`modules/evaluator.py`)
`BaseEvaluator` defines the interface; subclasses: `MMMUEvaluator`, `ScienceQAEvaluator`, `VideoMMEEvaluator`, `AirBenchEvaluator`. Each has:
- `calibrate()` — runs forward passes on calibration data to collect activation statistics into `QuantAct` modules (sets `_calibrate=True`, then computes dispersion scores)
- `prepare()` — computes outlier masks from dispersion scores, sets search mode (`set_search(True)`) for sparse-channel-aware quantization
- `evaluate()` — runs inference on the full test split, saves results

### 4. Quantization Layer Modules (`modules/`)
Each quantization scheme has its own module that replaces `nn.Linear` layers recursively:
- `hif8_layers.py` / `hif4_layers.py` — HiF8/HiF4 floating-point simulation (GPU CUDA + NPU AscendC backends under `HiF8_NPU_GPU_simulator/` and `HiFloat4-main/`)
- `smoothquant_layers.py` — SmoothQuant W8A8/W4A4/W4A16 simulation
- `awq_layers.py` — AWQ with per-group weight quantization
- `mbq_layers.py` — Modality-aware MBQ (separate calibration per modality: text vs multimodal)
- `mquant_layers.py` — MQuant W4A4 static quantization
- `quarot_layers.py` — QuaRot with Hadamard rotation + RTN pseudo-quantization
- `rtn_layers.py` — Round-to-nearest weight quantization

### 5. Custom bitsandbytes (`bnb_src/`)
The `MorphoQuantActWrapper` depends on `bitsandbytes.quantization_utils.quant_modules.QuantAct`, which lives in the locally modified bitsandbytes at `bnb_src/`. This must be installed/importable (typically via `PYTHONPATH` or editable install). The standard bitsandbytes lacks Morpho-specific features (gamma parameters, dispersion-based outlier detection, sparse channel compensation).

## Entry Point Pattern

Each `wy_inference_<benchmark>.py` follows the same structure:
1. Parse `--config` argument
2. `AppConfig.from_yaml(args.config)`
3. Optionally set up HiF8/HiF4 import paths and QuantAct debug mode
4. `ModelBuilder.build(config)` → `(model, processor)`
5. Optionally register activation/attention hooks via `visualizer.py`
6. Instantiate the benchmark-specific `Evaluator`
7. Call `evaluator.calibrate()`, `evaluator.prepare()`, `evaluator.evaluate()`

## Config Hierarchy

Configs are organized as `configs/<model>/<method>/<benchmark>_<method>.yaml`:
- `configs/qwen2.5-omni-7b/` — all methods for 7B model
- `configs/qwen2.5-omni-3b/` — 3B model configs
- `configs/internvl2.5-8b/` — InternVL2.5 configs

## Key Path Conventions

All paths use `/private/wy/` as the base:
- Pretrained models: `/private/wy/pretrained_models/`
- Datasets: `/private/wy/datasets/`
- Logs/outputs: `/private/wy/logs/MorphoQuant/`
- The conda environment is `MorphoQuant` (see `environment.yml`)
