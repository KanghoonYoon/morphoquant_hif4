#!/usr/bin/env bash
# ==============================================================================
# MorphoQuant 统一推理入口
# 将 MMMU / ScienceQA / AIR-Bench / Video-MME 四套推理流程合并为单一脚本，
# 通过子命令选择基准，支持批量配置文件与常见环境变量覆盖。
#
# 用法:
#   ./wy_inference.sh <benchmark> [benchmark ...]
#   ./wy_inference.sh all
#
# 环境变量（可选）:
#   BNB_CUDA_VERSION       默认 128
#   DEFAULT_GPU            未设置 CUDA_VISIBLE_DEVICES 且未使用 --gpu 时的默认 GPU，默认 4
#   CUDA_VISIBLE_DEVICES   若已在启动本脚本前导出，则对所有子任务沿用（除非使用 --gpu）
#   PYTHON                 默认 python
#
# 示例:
#   ./wy_inference.sh scienceqa
#   ./wy_inference.sh mmmu videomme
#   ./wy_inference.sh --gpu 0 all
#   DEFAULT_GPU=5 ./wy_inference.sh airbench
#   ./wy_inference.sh all --dry-run
# ==============================================================================

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

readonly PYTHON="${PYTHON:-python}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-124}"

# 若用户在调用前已导出 CUDA_VISIBLE_DEVICES，则记为“全局 GPU 偏好”
USER_CUDA_WAS_SET=false
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  USER_CUDA_WAS_SET=true
  readonly INITIAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

GPU_CLI_OVERRIDE=""
DRY_RUN=false
# 未使用 --gpu、且调用前未导出 CUDA_VISIBLE_DEVICES 时使用的默认 GPU
DEFAULT_GPU="${DEFAULT_GPU:-4}"

usage() {
  cat <<'USAGE'
用法: wy_inference.sh [选项] <子命令> [子命令 ...]

选项与子命令可任意穿插（例如: ./wy_inference.sh mmmu --gpu 0）。

子命令:
  mmmu       MMMU 推理（wy_inference_mmmu.py）
  scienceqa  ScienceQA 推理（wy_inference_scienceqa.py）
  airbench   AIR-Bench 推理（wy_inference_airbench.py）
  videomme   Video-MME 推理（wy_inference_videomme.py）
  all        按顺序运行以上四个基准

选项:
  -h, --help       显示本说明
  -g, --gpu ID     指定所有任务使用的 GPU（最高优先级）
  -n, --dry-run    仅打印将执行的命令，不实际运行

GPU 优先级（从高到低）:
  1) -g / --gpu
  2) 启动本脚本前已导出的 CUDA_VISIBLE_DEVICES
  3) 环境变量 DEFAULT_GPU（默认值为 4）
USAGE
}

log_sep() {
  echo "=================================================="
}

resolve_gpu() {
  if [[ -n "${GPU_CLI_OVERRIDE}" ]]; then
    echo "${GPU_CLI_OVERRIDE}"
    return
  fi
  if [[ "${USER_CUDA_WAS_SET}" == true ]]; then
    echo "${INITIAL_CUDA_VISIBLE_DEVICES}"
    return
  fi
  echo "${DEFAULT_GPU}"
}

run_one_config() {
  local bench="$1"
  local config_file="$2"
  local py_script="$3"
  local gpu
  gpu="$(resolve_gpu)"

  log_sep
  echo "基准: ${bench}"
  echo "GPU (CUDA_VISIBLE_DEVICES): ${gpu}"
  echo "配置文件: ${config_file}"
  log_sep

  export CUDA_VISIBLE_DEVICES="${gpu}"

  local -a cmd=("${PYTHON}" "${SCRIPT_DIR}/${py_script}" --config "${config_file}")
  if [[ "${DRY_RUN}" == true ]]; then
    printf '[dry-run] '; printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  "${cmd[@]}"
  echo "${bench} 测试完成 (${config_file})。"
  echo ""
}

configs_for_benchmark() {
  case "$1" in
    mmmu)
      printf '%s\n' "configs/internvl2.5-8b/morpho_withhif/mmmu_morpho_withhif4.yaml"
      # 其他可选配置（按需取消注释后移入上方列表）:
      # "configs/qwen2.5-omni-7b/fp16/mmmu_fp16.yaml"
      # "configs/qwen2.5-omni-7b/mquant/mmmu_mquant_w4a4.yaml"
      # "configs/qwen2.5-omni-7b/smoothquant/mmmu_smoothquant_w4a16.yaml"
      # "configs/qwen2.5-omni-3b/qlora/mmmu_qlora_w4a16.yaml"
      # "configs/qwen2.5-omni-7b/qlora/mmmu_qlora_w4a16.yaml"
      # "configs/internvl2.5-8b/qlora/mmmu_qlora_w4a16.yaml"
      ;;
    scienceqa)
      printf '%s\n' "configs/qwen2.5-omni-7b/qlora/scienceqa_qlora_w4a16.yaml"
      # "configs/qwen2.5-omni-7b/fp16/scienceqa_fp16.yaml"
      # "configs/qwen2.5-omni-7b/smoothquant/scienceqa_smoothquant_w4a16.yaml"
      # "configs/qwen2.5-omni-3b/qlora/scienceqa_qlora_w4a16.yaml"
      # "configs/qwen2.5-omni-7b/qlora/scienceqa_qlora_w4a16.yaml"
      # "configs/internvl2.5-8b/qlora/scienceqa_qlora_w4a16.yaml"
      ;;
    airbench)
      printf '%s\n' "configs/qwen2.5-omni-7b/qlora/airbench_qlora_w4a16.yaml"
      # "configs/qwen2.5-omni-7b/fp16/airbench_fp16.yaml"
      # "configs/qwen2.5-omni-7b/smoothquant/airbench_smoothquant_w4a16.yaml"
      # "configs/qwen2.5-omni-3b/qlora/airbench_qlora_w4a16.yaml"
      # "configs/qwen2.5-omni-7b/qlora/airbench_qlora_w4a16.yaml"
      # "configs/internvl2.5-8b/qlora/airbench_qlora_w4a16.yaml"
      ;;
    videomme)
      printf '%s\n' "configs/qwen2.5-omni-7b/morpho_withhif/videomme_morpho_withhif4.yaml"
      # "configs/qwen2.5-omni-7b/fp16/videomme_fp16.yaml"
      # "configs/qwen2.5-omni-7b/smoothquant/videomme_smoothquant_w4a16.yaml"
      # "configs/qwen2.5-omni-3b/qlora/videomme_qlora_w4a16.yaml"
      # "configs/qwen2.5-omni-7b/qlora/videomme_qlora_w4a16.yaml"
      # "configs/internvl2.5-8b/qlora/videomme_qlora_w4a16.yaml"
      ;;
    *)
      echo "未知基准: $1" >&2
      return 1
      ;;
  esac
}

py_script_for_benchmark() {
  case "$1" in
    mmmu) echo "wy_inference_mmmu.py" ;;
    scienceqa) echo "wy_inference_scienceqa.py" ;;
    airbench) echo "wy_inference_airbench.py" ;;
    videomme) echo "wy_inference_videomme.py" ;;
    *)
      echo "未知基准: $1" >&2
      return 1
      ;;
  esac
}

run_benchmark() {
  local bench="$1"
  local py_script
  py_script="$(py_script_for_benchmark "${bench}")" || return 1

  local config_file
  while IFS= read -r config_file || [[ -n "${config_file}" ]]; do
    [[ -z "${config_file}" ]] && continue
    [[ "${config_file}" =~ ^[[:space:]]*# ]] && continue
    config_file="$(echo "${config_file}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "${config_file}" ]] && continue
    run_one_config "${bench}" "${config_file}" "${py_script}"
  done < <(configs_for_benchmark "${bench}")
}

expand_all_benchmarks() {
  printf '%s\n' mmmu scienceqa airbench videomme
}

SELECTED_BENCHES=()

parse_args() {
  local -a positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      -g|--gpu)
        [[ $# -ge 2 ]] || { echo "选项 $1 需要参数" >&2; exit 1; }
        GPU_CLI_OVERRIDE="$2"
        shift 2
        ;;
      -n|--dry-run)
        DRY_RUN=true
        shift
        ;;
      --)
        shift
        positional+=("$@")
        break
        ;;
      -*)
        echo "未知选项: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done
  if [[ ${#positional[@]} -eq 0 ]]; then
    usage >&2
    exit 1
  fi
  SELECTED_BENCHES=("${positional[@]}")
}

main() {
  parse_args "$@"

  local b
  for b in "${SELECTED_BENCHES[@]}"; do
    case "${b}" in
      all)
        local sub
        while IFS= read -r sub; do
          run_benchmark "${sub}"
        done < <(expand_all_benchmarks)
        ;;
      mmmu|scienceqa|airbench|videomme)
        run_benchmark "${b}"
        ;;
      *)
        echo "无效子命令: ${b}" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
}

main "$@"
