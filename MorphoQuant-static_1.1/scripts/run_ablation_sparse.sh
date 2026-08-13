#!/usr/bin/env bash
# ==============================================================================
# MorphoQuant Ablation: Sparse Compensation × Loss Function
#
# 4 groups, 2 GPUs in parallel:
#   GPU 4: Group 1 (no sparse + L2) → Group 3 (sparse + L3)
#   GPU 5: Group 2 (sparse + L2)     → Group 4 (sparse + L3+cosine)
#
# Each group runs MMMU first, then ScienceQA, on qwen2.5-omni-3B.
#
# Usage:
#   conda activate MorphoQuant
#   ./scripts/run_ablation_sparse.sh
#   ./scripts/run_ablation_sparse.sh --dry-run
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

# Group 1: No sparse compensation + L2 loss
readonly G1_MMMU="${ABL}/mmmu_morpho_no_sparse_l2.yaml"
readonly G1_SCIQA="${ABL}/scienceqa_morpho_no_sparse_l2.yaml"

# Group 2: Sparse compensation + L2 loss
readonly G2_MMMU="${ABL}/mmmu_morpho_sparse_l2.yaml"
readonly G2_SCIQA="${ABL}/scienceqa_morpho_sparse_l2.yaml"

# Group 3: Sparse compensation + L3.0 loss
readonly G3_MMMU="${ABL}/mmmu_morpho_sparse_l3.yaml"
readonly G3_SCIQA="${ABL}/scienceqa_morpho_sparse_l3.yaml"

# Group 4: Sparse compensation + L3.0 loss + Cosine
readonly G4_MMMU="${ABL}/mmmu_morpho_sparse_l3_cosine.yaml"
readonly G4_SCIQA="${ABL}/scienceqa_morpho_sparse_l3_cosine.yaml"

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
readonly LOG_DIR="/private/wy/logs/MorphoQuant/ablation"

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

run_group() {
    local label="$1" gpu="$2" cfg_mmmu="$3" cfg_sciqa="$4" log_subdir="$5"

    local d="${LOG_DIR}/${log_subdir}"
    mkdir -p "${d}/mmmu_results" "${d}/scienceqa_results"

    log_sep
    echo "[$(date '+%H:%M:%S')] === ${label} (GPU ${gpu}) ==="
    log_sep

    run_one "${label}/MMMU"     "${gpu}" "${cfg_mmmu}"  "wy_inference_mmmu.py"      "${d}/mmmu.log" || true
    run_one "${label}/ScienceQA" "${gpu}" "${cfg_sciqa}" "wy_inference_scienceqa.py" "${d}/sciqa.log" || true

    log_sep
    echo "[$(date '+%H:%M:%S')] === ${label} 全部完成 ==="
    log_sep
}

print_result_summary() {
    echo ""
    log_sep
    echo "  消融实验结果摘要"
    log_sep
    echo ""

    for group in no_sparse_l2 sparse_l2 sparse_l3 sparse_l3_cosine; do
        echo "--- ${group} ---"
        for bench in mmmu scienceqa; do
            local rf="${LOG_DIR}/${group}/${bench}_results/summary_report.csv"
            local lf="${LOG_DIR}/${group}/${bench}.log"
            if [[ -f "${rf}" ]]; then
                echo "  ${bench}: $(head -3 "${rf}" | tail -1)"
            elif [[ -f "${lf}" ]]; then
                echo "  ${bench}: (no summary — check ${lf})"
            else
                echo "  ${bench}: (no output)"
            fi
        done
        echo ""
    done
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
                echo "Runs 4-group ablation in parallel across GPUs 4 and 5."
                echo "  GPU 4: Group 1 (no sparse, L2) → Group 3 (sparse, L3)"
                echo "  GPU 5: Group 2 (sparse, L2)   → Group 4 (sparse, L3+cosine)"
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
    echo "  MorphoQuant Ablation: 4 Groups (Sparse Comp × Loss Fn)"
    echo "  Model: Qwen2.5-Omni-3B"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    printf "\n"
    printf "  %-30s  %-6s  %s\n" "Group" "GPU" "Config"
    printf "  %-30s  %-6s  %s\n" "------------------------------" "------" "-------"
    printf "  %-30s  %-6s  %s\n" "G1: No Sparse + L2"         "4"      "no_sparse_l2"
    printf "  %-30s  %-6s  %s\n" "G2: Sparse + L2"           "5"      "sparse_l2"
    printf "  %-30s  %-6s  %s\n" "G3: Sparse + L3"           "4"      "sparse_l3"
    printf "  %-30s  %-6s  %s\n" "G4: Sparse + L3+Cosine"   "5"      "sparse_l3_cosine"
    echo ""

    mkdir -p "${LOG_DIR}"

    if [[ "${DRY_RUN}" == true ]]; then
        echo "[DRY RUN] 仅预览命令，不执行。"
        echo ""
    fi

    # --- GPU 4 pipeline: G1 → G3 (background) ------------------------------
    (
        run_group "G1-NoSparse-L2" "4" "${G1_MMMU}" "${G1_SCIQA}" "no_sparse_l2"
        run_group "G3-Sparse-L3"   "4" "${G3_MMMU}" "${G3_SCIQA}" "sparse_l3"
    ) &
    local pid_g4=$!

    # --- GPU 5 pipeline: G2 → G4 (background) ------------------------------
    (
        run_group "G2-Sparse-L2"        "5" "${G2_MMMU}" "${G2_SCIQA}" "sparse_l2"
        run_group "G4-Sparse-L3-Cosine" "5" "${G4_MMMU}" "${G4_SCIQA}" "sparse_l3_cosine"
    ) &
    local pid_g5=$!

    echo ""
    echo "后台任务已启动:"
    echo "  GPU 4 pipeline: PID ${pid_g4}"
    echo "  GPU 5 pipeline: PID ${pid_g5}"
    echo ""
    echo "等待完成..."

    local rc4=0 rc5=0
    wait ${pid_g4} || rc4=$?
    wait ${pid_g5} || rc5=$?

    echo ""
    echo "============================================================"
    echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  GPU 4 exit: ${rc4}   GPU 5 exit: ${rc5}"
    echo "============================================================"

    print_result_summary
}

main "$@"
