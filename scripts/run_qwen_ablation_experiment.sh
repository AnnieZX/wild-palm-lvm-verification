#!/usr/bin/env bash
# Orchestrate the full Qwen2.5-VL verification ablation experiment (A1–A5).
#
# Called by jobs/run_qwen_ablation.slurm or directly from the project root:
#   SAMPLE_SIZE=500 EXPERIMENT_ID=20260707_0100 bash scripts/run_qwen_ablation_experiment.sh
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
SAMPLE_SIZE="${SAMPLE_SIZE:-300}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SKIP_DATASET_PREP="${SKIP_DATASET_PREP:-0}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/outputs/verification_dataset}"
ABLATION_INPUTS_DIR="${ABLATION_INPUTS_DIR:-${PROJECT_DIR}/outputs/verification_ablation_${SAMPLE_SIZE}}"
VERIFICATION_ROOT="${PROJECT_DIR}/outputs/verification/qwen/${EXPERIMENT_ID}"
EVALUATION_ROOT="${PROJECT_DIR}/outputs/evaluation/qwen/${EXPERIMENT_ID}"

ABLATION_CODES=(A1 A2 A3 A4 A5)

EXPERIMENT_START=$(date +%s)
EXPERIMENT_START_DISPLAY=$(date)

echo "Qwen2.5-VL ablation experiment"
echo "  Project:          ${PROJECT_DIR}"
echo "  Sample size:      ${SAMPLE_SIZE}"
echo "  Experiment ID:    ${EXPERIMENT_ID}"
echo "  Verification out: ${VERIFICATION_ROOT}"
echo "  Evaluation out:   ${EVALUATION_ROOT}"
echo "  Started:          ${EXPERIMENT_START_DISPLAY}"
echo

# ---------------------------------------------------------------------------
# Step 0: Prepare verification dataset and ablation inputs
# ---------------------------------------------------------------------------
if [[ "${SKIP_DATASET_PREP}" != "1" ]]; then
    echo "=== Preparing verification dataset ==="
    python scripts/pipeline/generate_verification_dataset.py \
        --output-dir "${DATASET_DIR}"

    echo
    echo "=== Building ablation prompts (sample_count=${SAMPLE_SIZE}) ==="
    python scripts/pipeline/build_ablation_verification_prompts.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${ABLATION_INPUTS_DIR}" \
        --sample-count "${SAMPLE_SIZE}"
    echo
else
    echo "=== Skipping dataset preparation (SKIP_DATASET_PREP=1) ==="
    echo
fi

# ---------------------------------------------------------------------------
# Steps 1–5: Inference + evaluation per ablation
# ---------------------------------------------------------------------------
COMPLETED_CODES=()

for CODE in "${ABLATION_CODES[@]}"; do
    ABLATION_START=$(date +%s)
    ABLATION_START_DISPLAY=$(date)

    echo "===================================="
    echo "Running ${CODE}..."
    echo "Start time: ${ABLATION_START_DISPLAY}"
    echo "===================================="

    RESULTS_DIR="${VERIFICATION_ROOT}/${CODE}"
    EVAL_DIR="${EVALUATION_ROOT}/${CODE}"
    mkdir -p "${RESULTS_DIR}" "${EVAL_DIR}"

    python scripts/run_ablation_verification.py \
        --condition "${CODE}" \
        --ablation-dir "${ABLATION_INPUTS_DIR}" \
        --results-dir "${RESULTS_DIR}" \
        --limit "${SAMPLE_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --model-config "${MODEL_CONFIG}"

    python scripts/evaluate_verification_against_groundtruth.py \
        --results-dir "${RESULTS_DIR}" \
        --index-csv "${DATASET_DIR}/index.csv" \
        --output-dir "${EVAL_DIR}" \
        --condition-code "${CODE}"

    python scripts/compute_verification_metrics.py \
        --evaluation-dir "${EVAL_DIR}"

    ABLATION_END=$(date +%s)
    ELAPSED=$((ABLATION_END - ABLATION_START))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))

    echo
    echo "Completed ${CODE}"
    echo "Elapsed time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo

    COMPLETED_CODES+=("${CODE}")
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
EXPERIMENT_END=$(date +%s)
TOTAL_ELAPSED=$((EXPERIMENT_END - EXPERIMENT_START))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo "===================================="
echo "Experiment Complete"
echo "===================================="
for CODE in "${COMPLETED_CODES[@]}"; do
    echo "✔ ${CODE}"
done
echo
echo "Experiment ID:  ${EXPERIMENT_ID}"
echo "Sample size:    ${SAMPLE_SIZE}"
echo "Total runtime:  ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "Verification:   ${VERIFICATION_ROOT}"
echo "Evaluation:     ${EVALUATION_ROOT}"
