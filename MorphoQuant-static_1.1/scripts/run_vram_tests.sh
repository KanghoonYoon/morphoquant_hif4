#!/usr/bin/env bash
# ==============================================================================
# MorphoQuant VRAM Inference Memory Test — Batch Runner
#
# Runs all 22 quantization configs on GPU 6, records per-config VRAM stats,
# and generates a final markdown summary table.
#
# Usage:
#   bash scripts/run_vram_tests.sh
#   bash scripts/run_vram_tests.sh --dry-run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-124}"
export HF_ENDPOINT="https://hf-mirror.com"

PYTHON="${PYTHON:-python}"
# Use conda run if MorphoQuant environment exists
if conda env list 2>/dev/null | grep -q "^MorphoQuant "; then
    PYTHON="conda run -n MorphoQuant --no-capture-output python"
fi
NUM_SAMPLES="${NUM_SAMPLES:-5}"
RESULTS_DIR="${SCRIPT_DIR}/vram_results"
RESULTS_JSONL="${RESULTS_DIR}/all_results.jsonl"
RESULTS_MD="${RESULTS_DIR}/vram_report.md"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] Would run VRAM tests on GPU ${CUDA_VISIBLE_DEVICES}"
fi

mkdir -p "${RESULTS_DIR}"

# ==============================================================================
# Config list: "model_label|config_path"
# ==============================================================================
CONFIGS=(
    # === qwen2.5-omni-3B (10 configs) ===
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/smoothquant/scienceqa_smoothquant_w4a16.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/awq/scienceqa_awq_w4a16.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/mbq/scienceqa_mbq_w8a8.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/mquant/scienceqa_mquant_w8a8.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/morpho/scienceqa_morpho_w8a8.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/morpho_withhif/scienceqa_morpho_withhif8.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/mbq/scienceqa_mbq_w4a4.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/mquant/scienceqa_mquant_w4a4.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/morpho/scienceqa_morpho.yaml"
    "Qwen2.5-Omni-3B|configs/qwen2.5-omni-3b/morpho_withhif/scienceqa_morpho_withhif4.yaml"

    # === qwen2.5-omni-7B (7 configs) ===
    "Qwen2.5-Omni-7B|configs/qwen2.5-omni-7b/qlora/scienceqa_qlora_w4a16.yaml"
    "Qwen2.5-Omni-7B|configs/qwen2.5-omni-7b/smoothquant/scienceqa_smoothquant_w4a16.yaml"
    "Qwen2.5-Omni-7B|configs/qwen2.5-omni-7b/awq/scienceqa_awq_w4a16.yaml"
    "Qwen2.5-Omni-7B|configs/qwen2.5-omni-7b/mbq/scienceqa_mbq_w4a4.yaml"
    "Qwen2.5-Omni-7B|configs/qwen2.5-omni-7b/mquant/scienceqa_mquant_w4a4.yaml"
    "Qwen2.5-Omni-7B|configs/qwen2.5-omni-7b/morpho/scienceqa_morpho.yaml"
    "Qwen2.5-Omni-7B|configs/qwen2.5-omni-7b/morpho_withhif/scienceqa_morpho_withhif4.yaml"

    # === internvl2.5-8B (5 configs) ===
    "InternVL2.5-8B|configs/internvl2.5-8b/mbq/scienceqa_mbq_w4a4.yaml"
    "InternVL2.5-8B|configs/internvl2.5-8b/quarot/scienceqa_quarot_w4a4.yaml"
    "InternVL2.5-8B|configs/internvl2.5-8b/freeact/scienceqa_freeact_w4a4.yaml"
    "InternVL2.5-8B|configs/internvl2.5-8b/morpho/scienceqa_morpho.yaml"
    "InternVL2.5-8B|configs/internvl2.5-8b/morpho_withhif/scienceqa_morpho_withhif4.yaml"
)

TOTAL=${#CONFIGS[@]}

# ==============================================================================
# Functions
# ==============================================================================

log_sep() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

# ==============================================================================
# Main
# ==============================================================================

log_sep "MorphoQuant VRAM Test — ${TOTAL} configs on GPU ${CUDA_VISIBLE_DEVICES}"
echo "Num inference samples per config: ${NUM_SAMPLES}"
echo "Results dir: ${RESULTS_DIR}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Clear previous results
> "${RESULTS_JSONL}"

SUCCESS=0
FAILED=0
START_TIME=$(date +%s)

for idx in "${!CONFIGS[@]}"; do
    IFS='|' read -r MODEL_LABEL CONFIG_PATH <<< "${CONFIGS[$idx]}"
    CURRENT=$((idx + 1))

    log_sep "[${CURRENT}/${TOTAL}] ${MODEL_LABEL} — ${CONFIG_PATH}"

    if [[ ! -f "${PROJECT_DIR}/${CONFIG_PATH}" ]]; then
        echo "[SKIP] Config file not found: ${CONFIG_PATH}"
        echo "{\"config_path\":\"${CONFIG_PATH}\",\"model\":\"${MODEL_LABEL}\",\"status\":\"config_missing\",\"error\":\"File not found\"}" >> "${RESULTS_JSONL}"
        FAILED=$((FAILED + 1))
        continue
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        echo "[DRY-RUN] ${PYTHON} scripts/test_vram.py --config ${CONFIG_PATH} --num_samples ${NUM_SAMPLES}"
        continue
    fi

    CONFIG_START=$(date +%s)

    set +e
    ${PYTHON} "${SCRIPT_DIR}/test_vram.py" \
        --config "${CONFIG_PATH}" \
        --num_samples "${NUM_SAMPLES}" \
        2>&1 | tee "${RESULTS_DIR}/log_${CURRENT}.txt"
    EXIT_CODE=$?
    set -e

    CONFIG_END=$(date +%s)
    CONFIG_ELAPSED=$((CONFIG_END - CONFIG_START))
    CONFIG_MIN=$((CONFIG_ELAPSED / 60))
    CONFIG_SEC=$((CONFIG_ELAPSED % 60))

    # Extract JSON result line from the log
    RESULT_LINE=$(grep "^__RESULT_JSON__" "${RESULTS_DIR}/log_${CURRENT}.txt" | head -1 | sed 's/^__RESULT_JSON__ //' || true)

    if [[ -n "${RESULT_LINE}" ]]; then
        echo "${RESULT_LINE}" >> "${RESULTS_JSONL}"
        STATUS=$(echo "${RESULT_LINE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "parse_error")
        if [[ "${STATUS}" == "ok" ]]; then
            SUCCESS=$((SUCCESS + 1))
            echo "[OK] ${MODEL_LABEL} — took ${CONFIG_MIN}m${CONFIG_SEC}s"
        else
            FAILED=$((FAILED + 1))
            echo "[FAIL] ${MODEL_LABEL} (status=${STATUS}) — took ${CONFIG_MIN}m${CONFIG_SEC}s"
        fi
    else
        # No valid JSON result found — create an error entry
        echo "{\"config_path\":\"${CONFIG_PATH}\",\"model\":\"${MODEL_LABEL}\",\"status\":\"no_result\",\"error\":\"No JSON result extracted\"}" >> "${RESULTS_JSONL}"
        FAILED=$((FAILED + 1))
        if [[ ${EXIT_CODE} -ne 0 ]]; then
            echo "[FAIL] ${MODEL_LABEL} — exit code ${EXIT_CODE}, no result JSON"
        else
            echo "[FAIL] ${MODEL_LABEL} — no result JSON found in output"
        fi
    fi
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

# ==============================================================================
# Generate summary table
# ==============================================================================

log_sep "Generating Report"
echo "Total time: ${TOTAL_MIN}m${TOTAL_SEC}s"
echo "Success: ${SUCCESS} / ${TOTAL}"
echo "Failed:  ${FAILED} / ${TOTAL}"

# Write report generator script
cat > /tmp/_gen_vram_report.py << 'PYEOF_REPORT'
import json, sys

results_file = sys.argv[1]
md_file = sys.argv[2]

results = []
with open(results_file) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

if not results:
    print("[WARN] No results to generate report from.")
    sys.exit(0)

# Separate by model
models_order = ["Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B", "InternVL2.5-8B"]
model_groups = {m: [] for m in models_order}
for r in results:
    model = r.get("model", "Unknown")
    if "3B" in model or "3b" in model:
        model_groups["Qwen2.5-Omni-3B"].append(r)
    elif "7B" in model or "7b" in model:
        model_groups["Qwen2.5-Omni-7B"].append(r)
    elif "InternVL" in model or "internvl" in model:
        model_groups["InternVL2.5-8B"].append(r)

def format_gb(val):
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"

# Build markdown table
lines = []
lines.append("# MorphoQuant VRAM Inference Memory Test Results\n")
lines.append(f"**Date**: auto-generated  ")
lines.append(f"**GPU**: NVIDIA (CUDA_VISIBLE_DEVICES=6)  ")
lines.append(f"**Dataset**: ScienceQA (test split, 5 inference samples per config)  \n")
lines.append("## Summary\n")

for model_name in models_order:
    group = model_groups[model_name]
    if not group:
        continue

    lines.append(f"### {model_name}\n")
    lines.append("| # | Method | Precision | Status | Model Load (GB) | Calibrate Peak (GB) | Prepare Peak (GB) | Inference Peak (GB) | Total Peak (GB) |")
    lines.append("|---|--------|-----------|--------|-----------------|---------------------|-------------------|---------------------|-----------------|")

    for i, r in enumerate(group, 1):
        status_icon = "✅" if r.get("status") == "ok" else "❌"
        method = r.get("method", "?")
        precision = r.get("precision", "?")
        model_load = format_gb(r.get("model_load_vram_gb", 0))
        calibrate = format_gb(r.get("calibrate_peak_vram_gb", 0))
        prepare = format_gb(r.get("prepare_peak_vram_gb", 0))
        inference = format_gb(r.get("inference_peak_vram_gb", 0))
        total = format_gb(r.get("total_peak_vram_gb", 0))

        row = f"| {i} | {method} | {precision} | {status_icon} | {model_load} | {calibrate} | {prepare} | {inference} | {total} |"
        lines.append(row)

        if r.get("status") != "ok":
            lines.append(f"| | | _{r.get('error', 'unknown error')[:80]}_ | | | | | | |")

    lines.append("")

# Add overall comparison table
lines.append("## Overall Comparison (Inference Peak VRAM)\n")
lines.append("| # | Model | Method | Precision | Inference Peak (GB) | Status |")
lines.append("|---|-------|--------|-----------|---------------------|--------|")

all_sorted = sorted(results, key=lambda r: (
    models_order.index(r.get("model", "Unknown")) if r.get("model", "Unknown") in models_order else 99,
    r.get("method", ""),
    r.get("precision", "")
))

for i, r in enumerate(all_sorted, 1):
    status_icon = "✅" if r.get("status") == "ok" else "❌"
    model = r.get("model", "?")
    method = r.get("method", "?")
    precision = r.get("precision", "?")
    inference = format_gb(r.get("inference_peak_vram_gb", 0))
    row = f"| {i} | {model} | {method} | {precision} | {inference} | {status_icon} |"
    lines.append(row)

# Write
content = "\n".join(lines)
with open(md_file, "w") as f:
    f.write(content)
print(content)
PYEOF_REPORT

python3 /tmp/_gen_vram_report.py "${RESULTS_JSONL}" "${RESULTS_MD}"
rm -f /tmp/_gen_vram_report.py

log_sep "Done"
echo "Report saved to: ${RESULTS_MD}"
echo "Raw results: ${RESULTS_JSONL}"
echo ""
echo "Summary: ${SUCCESS}/${TOTAL} successful, ${FAILED} failed."
echo "Total elapsed: ${TOTAL_MIN}m${TOTAL_SEC}s"
