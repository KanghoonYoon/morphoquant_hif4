#!/usr/bin/env bash
# ==============================================================================
# MorphoQuant Ablation: Sparse Compensation WITHOUT Boundary Co-optimization
#
# Group 1 (the only group): Keep sparse compensation, DISABLE boundary search.
# Runs MMMU on GPU 4 and ScienceQA on GPU 5 in parallel.
#
# Usage:
#   conda activate MorphoQuant
#   ./scripts/run_ablation_boundary.sh
#   ./scripts/run_ablation_boundary.sh --dry-run
# ==============================================================================

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-124}"
export CONDA_ENV="${CONDA_ENV:-MorphoQuant}"
export PYTHON="${PYTHON:-python}"

# Use conda run if not already in the right env
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    PYTHON="conda run -n ${CONDA_ENV} python"
fi

if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${PROJECT_DIR}/bnb_src"
else
    export PYTHONPATH="${PROJECT_DIR}/bnb_src:${PYTHONPATH}"
fi

DRY_RUN=false

# ---------------------------------------------------------------------------
# Config paths (relative to PROJECT_DIR)
# ---------------------------------------------------------------------------
readonly ABL="configs/qwen2.5-omni-3b/morpho/ablation"

# Group 1: Sparse compensation ON, Boundary co-optimization OFF
readonly CFG_MMMU="${ABL}/mmmu_morpho_sparse_noboundary.yaml"
readonly CFG_SCIQA="${ABL}/scienceqa_morpho_sparse_noboundary.yaml"

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
readonly LOG_DIR="/private/wy/logs/MorphoQuant/ablation"
readonly LOG_SUBDIR="sparse_noboundary"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log_sep() { echo "============================================================"; }

run_one() {
    local label="$1" gpu="$2" config="$3" py_script="$4" log_file="$5"

    echo "[$(date '+%H:%M:%S')] [${label}] 开始: ${py_script}"
    echo "  GPU: ${gpu}  |  Config: ${config}"

    if [[ "${DRY_RUN}" == true ]]; then
        echo "[dry-run] CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${PROJECT_DIR}/${py_script} --config ${PROJECT_DIR}/${config}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${PROJECT_DIR}/${py_script}" \
        --config "${PROJECT_DIR}/${config}" \
        > "${log_file}" 2>&1

    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] [${label}] 完成 ✓"
    else
        echo "[$(date '+%H:%M:%S')] [${label}] 失败 ✗ (exit code: ${rc})"
    fi
    return ${rc}
}

print_result_summary() {
    echo ""
    log_sep
    echo "  消融实验结果摘要: Sparse ON, Boundary OFF"
    log_sep
    echo ""

    for bench in mmmu scienceqa; do
        local rf="${LOG_DIR}/${LOG_SUBDIR}/${bench}_results/summary_report.csv"
        local lf="${LOG_DIR}/${LOG_SUBDIR}/${bench}.log"
        if [[ -f "${rf}" ]]; then
            echo "  ${bench}: $(head -3 "${rf}" | tail -1)"
        elif [[ -f "${lf}" ]]; then
            echo "  ${bench}: (no summary — check ${lf})"
        else
            echo "  ${bench}: (no output)"
        fi
    done
    echo ""
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                echo "Usage: $0 [--dry-run]"
                echo ""
                echo "Runs Group 1 ablation (sparse ON, boundary co-opt OFF)."
                echo "  GPU 4: MMMU"
                echo "  GPU 5: ScienceQA"
                exit 0
                ;;
            -n|--dry-run)
                DRY_RUN=true; shift ;;
            *)  echo "未知参数: $1"; exit 1 ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"

    echo "============================================================"
    echo "  MorphoQuant Ablation: Sparse ON, Boundary Co-opt OFF"
    echo "  Model: Qwen2.5-Omni-3B"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    printf "\n"
    printf "  %-30s  %-6s  %s\n" "Benchmark" "GPU" "Config"
    printf "  %-30s  %-6s  %s\n" "------------------------------" "------" "-------"
    printf "  %-30s  %-6s  %s\n" "MMMU"    "4" "sparse_noboundary"
    printf "  %-30s  %-6s  %s\n" "ScienceQA" "5" "sparse_noboundary"
    echo ""

    local d="${LOG_DIR}/${LOG_SUBDIR}"
    mkdir -p "${d}/mmmu_results" "${d}/scienceqa_results"

    if [[ "${DRY_RUN}" == true ]]; then
        echo "[DRY RUN] 仅预览命令，不执行。"
        echo ""
    fi

    # --- GPU 4: MMMU (background) ---
    (
        run_one "MMMU" "4" "${CFG_MMMU}" "wy_inference_mmmu.py" "${d}/mmmu.log"
    ) &
    local pid_g4=$!

    # --- GPU 5: ScienceQA (background) ---
    (
        run_one "ScienceQA" "5" "${CFG_SCIQA}" "wy_inference_scienceqa.py" "${d}/sciqa.log"
    ) &
    local pid_g5=$!

    echo ""
    echo "后台任务已启动:"
    echo "  GPU 4 (MMMU):     PID ${pid_g4}"
    echo "  GPU 5 (ScienceQA): PID ${pid_g5}"
    echo ""
    echo "等待完成..."

    local rc4=0 rc5=0
    wait ${pid_g4} || rc4=$?
    wait ${pid_g5} || rc5=$?

    echo ""
    echo "============================================================"
    echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  GPU 4 (MMMU):     exit ${rc4}"
    echo "  GPU 5 (ScienceQA): exit ${rc5}"
    echo "============================================================"

    print_result_summary
}

main "$@"
