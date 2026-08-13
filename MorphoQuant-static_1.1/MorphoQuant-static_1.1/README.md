# MorphoQuant

**Morpho Activation Quantization** applied to multimodal models including **Qwen2.5-Omni** and **InternVL2.5**, powered by a locally modified **bitsandbytes** (`bnb_src/`) backend. This repository provides a unified pipeline for inference and evaluation across multiple benchmarks.

## Features

- **Unified Configuration** — `AppConfig` in `config.py` loads YAML-based settings organized into four groups: `model`, `quant`, `data`, and `run`.
- **Model Builder** — `ModelBuilder.build()` in `modules/model_factory.py` loads weights and processors by `model_type` / `quant_method`, injecting Morpho-related wrappers into linear layers (e.g., `MorphoQuantActWrapper`, `MorphoHiF8Linear`).
- **Benchmark Evaluation** — `modules/evaluator.py` provides per-dataset `Evaluator` classes (MMMU, ScienceQA, VideoMME, AIR-Bench), handling calibration, data preparation, generation, and result output.
- **Optional Visualization** — `visualizer.py` supports activation histograms and attention map hooking/plotting, controlled by YAML flags such as `apply_hooks` and `save_attention_maps`.
- **HiF8 Simulation** — When `quant.simulate_hif8` is enabled, the pipeline attempts to load `quant_cy` from `modules/HiF8_NPU_GPU_simulator/GPU_Cuda/hif8_cuda/`. The `morpho_withhif8` quant method combines Morpho and HiF8.

## Directory Structure (Core)

| Path | Description |
|------|-------------|
| `config.py` | Configuration dataclass and `from_yaml` loader |
| `configs/*.yaml` | Example YAML configurations for various tasks and model scales |
| `modules/model_factory.py` | Model building and quantization pathway setup |
| `modules/evaluator.py` | Dataset evaluation logic |
| `modules/hif8_layers.py` | HiF8 / Morpho-HiF8 layer replacements |
| `wy_inference_*.py` | Entry-point scripts for each benchmark |
| `bnb_src/` | Locally modified bitsandbytes source (coupled with this project) |
| `qwen-omni-utils/` | Qwen-Omni multimodal preprocessing utilities |

## Environment Setup

- A **Conda** environment is recommended (e.g., `conda activate omni`). Install the following dependencies:
  - **PyTorch** (with CUDA support)
  - `transformers`, `accelerate`, `datasets`
  - `Pillow`, `torchvision`, `scikit-learn`, `matplotlib`, `tqdm`
- The **`qwen_omni_utils`** package must be importable. Either install `qwen-omni-utils/` in editable mode or add it to `PYTHONPATH`.
- **bitsandbytes** — Use the locally modified source under `bnb_src/` so that Morpho-related modules inside `bitsandbytes.quantization_utils.quant_modules` are available.

## Quick Start

All entry points accept a YAML configuration file via `--config`. Update `model_path`, `dataset_dir`, `save_dir`, and `data_root` in `configs/` to match your local paths before running.

```bash
# MMMU
python wy_inference_mmmu.py --config configs/qwen2.5-omni-3b/morpho/mmmu_morpho.yaml

# ScienceQA
python wy_inference_scienceqa.py --config configs/qwen2.5-omni-3b/morpho/scienceqa_morpho.yaml

# VideoMME
python wy_inference_videomme.py --config configs/qwen2.5-omni-3b/morpho/videomme_morpho.yaml

# AIR-Bench (requires Foundation_meta.json under data.data_root)
python wy_inference_airbench.py --config configs/qwen2.5-omni-3b/morpho/airbench_morpho.yaml
```

The scripts default `HF_ENDPOINT` to `https://hf-mirror.com`. Switch to the official Hugging Face Hub or another mirror as needed.

## Configuration Reference

### `model`

| Key | Description |
|-----|-------------|
| `quant_method` | `bf16`, `int8`, `awq`, `hqq`, `morpho`, `qvlm`, `morpho_withhif8`, `qlora`, etc. (see `ModelBuilder` for full support) |
| `model_type` | Defaults to `qwen2_5_omni`; InternVL configs use `internvl2_5`, etc. |

### `quant`

| Key | Description |
|-----|-------------|
| `activation_bitwidth` | Bit-width for activation quantization |
| `gamma_inf` / `gamma_cos` | Morpho-specific hyperparameters |
| `simulate_hif8` | Enable HiF8 simulation path |

### `data`

| Key | Description |
|-----|-------------|
| `num_samples` | Number of samples (`-1` for full dataset) |
| `calib_size` / `calib_batch_size` | Calibration sample count and batch size |
| `target_subject` | Subject filter (e.g., for MMMU) |
| `dataset_dir` | Dataset directory |
| `output_file` | Results output file |
| `run_pca` | PCA visualization flag |

### `run`

| Key | Description |
|-----|-------------|
| `save_dir` | Output directory for results |
| `seed` | Random seed |

## Qwen2.5-Omni Submodule Layout

For cross-referencing with the source code:

- **Thinker**: `thinker.audio_tower`, `thinker.visual`, `thinker.model`
- **Talker**: `talker.model`
- **Token2Wav**

## Testing

A lightweight smoke test is available at `tests/smoke_morpho_internvl.py`. Run it after setting up dependencies to verify basic functionality.

## Development Notes

If `git push` fails due to SSH issues, reinitialize the SSH agent with your key and retry:

```bash
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_rsa_wy_github && git push
```

## License

This project is provided for research purposes. See the respective upstream repositories (bitsandbytes, Qwen, InternVL) for their applicable licenses.