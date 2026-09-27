#!/bin/bash
#SBATCH --job-name=ivl35hf_Ax_5747
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/internvl3_5_hf_A%a_5747_%j.out
#SBATCH --error=logs/slurm/internvl3_5_hf_A%a_5747_%j.err
#
# InternVL3.5-8B-HF canonical A1–A5 @5747 (full verification set).
# Submit only AFTER Stage 1 balanced-100 qualification PASS.
# Prefer validated L40S (do not silently migrate to H200):
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=20260924_internvl3_5_hf_A1A5_5747,ABLATION=A1 \
#     --job-name=ivl35hf_A1_5747 \
#     --output=logs/slurm/internvl3_5_hf_A1_5747_%j.out \
#     --error=logs/slurm/internvl3_5_hf_A1_5747_%j.err \
#     jobs/run_internvl3_5_hf_Ax_5747.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

ABLATION="${ABLATION:?Set ABLATION to A1, A2, A3, A4, or A5}"
case "${ABLATION}" in
    A1) CONDITION_DIR="A1_overlay_only" ;;
    A2) CONDITION_DIR="A2_overlay_confidence" ;;
    A3) CONDITION_DIR="A3_overlay_confidence_geometry" ;;
    A4) CONDITION_DIR="A4_overlay_crop_confidence" ;;
    A5) CONDITION_DIR="A5_crop_only" ;;
    *)
        echo "ERROR: ABLATION must be A1|A2|A3|A4|A5 (got ${ABLATION})" >&2
        exit 1
        ;;
esac

# Same runtime pattern as Stage 0 / Stage 1 qualification (default cluster python).
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_internvl3_5_hf_A1A5_5747}"
MODEL="${MODEL:-internvl3_5_hf}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/InternVL3_5-8B-HF}"
SAMPLE_SIZE=5747

PROMPT_INDEX="${PROJECT_DIR}/outputs/verification_ablation_5747/${CONDITION_DIR}/prompt_index.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
EVAL_DIR="${PROJECT_DIR}/outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
DATASET_INDEX="${PROJECT_DIR}/outputs/verification_dataset/index.csv"

mkdir -p logs/slurm "${RESULTS_DIR}" "${EVAL_DIR}"

echo "===== InternVL3.5-HF Canonical ${ABLATION}-5747 ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Ablation:      ${ABLATION} (${CONDITION_DIR})"
echo "Sample size:   ${SAMPLE_SIZE}"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Evaluation:    ${EVAL_DIR}"
echo "Model path:    ${MODEL_PATH}"
echo "Resume:        ${RESUME}"
echo "Start time:    $(date)"
echo

case "${RESULTS_DIR}" in
    */outputs/verification/qwen/*|*/outputs/verification/qwen2_5_vl/*|*/outputs/evaluation/qwen/*|\
    */outputs/verification/internvl3/20260909_*|*/outputs/evaluation/internvl3/20260909_*|\
    */outputs/verification/internvl3_5_hf/20260924_internvl3_5_hf_qual/*)
        echo "ERROR: refusing protected path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

echo "Python: $(which python)"
python --version
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from transformers import AutoModelForImageTextToText, AutoProcessor, InternVLForConditionalGeneration; print('InternVL HF classes OK')"
echo
echo "=== nvidia-smi ==="
nvidia-smi
echo

# Fail fast on GPUs with active uncorrectable ECC.
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,ecc.errors.uncorrected.volatile.total",
     "--format=csv,noheader,nounits"],
    text=True,
)
bad = []
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    idx, ecc = parts[0], parts[1]
    try:
        ecc_n = int(ecc)
    except ValueError:
        ecc_n = -1
    print(f"GPU {idx} volatile_uncorrected_ecc_total={ecc}")
    if ecc_n > 0:
        bad.append((idx, ecc_n))
if bad:
    print("ERROR: refusing GPU(s) with uncorrected ECC:", bad, file=sys.stderr)
    sys.exit(2)
print("ECC_GUARD_PASS")
PY

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -f "${MODEL_PATH}/DOWNLOAD_META.txt" ]]; then
    cat "${MODEL_PATH}/DOWNLOAD_META.txt"
fi
echo

# Full-set integrity + parity vs A1 @5747
python - <<PY
import pandas as pd
a1 = set(pd.read_csv("outputs/verification_ablation_5747/A1_overlay_only/prompt_index.csv").sample_id.astype(str))
idx = pd.read_csv("${PROMPT_INDEX}")
assert len(idx) == 5747 and idx.sample_id.nunique() == 5747
assert set(idx.sample_id.astype(str)) == a1
assert set(idx.condition.unique()) == {"${CONDITION_DIR}"}
print("PARITY_PASS: ${ABLATION} IDs identical to A1; N=5747")
PY

VERIFICATION_ARGS=(
    --model internvl3_5_hf
    --condition "${ABLATION}"
    --limit "${SAMPLE_SIZE}"
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-path "${MODEL_PATH}"
    --model-config "${PROJECT_DIR}/configs/models/internvl3_5_hf.yaml"
)

if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

echo "===== Inference (${ABLATION}-5747) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
echo "===== Canonical evaluation ====="
python -u scripts/evaluate_verification_against_groundtruth.py \
    --results-dir "${RESULTS_DIR}" \
    --index-csv "${DATASET_INDEX}" \
    --output-dir "${EVAL_DIR}" \
    --condition-code "${ABLATION}"

python -u scripts/compute_verification_metrics.py \
    --evaluation-dir "${EVAL_DIR}"

echo
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "Evaluation: ${EVAL_DIR}"
echo "${ABLATION}_5747_COMPLETE"
