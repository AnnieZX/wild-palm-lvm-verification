#!/bin/bash
#SBATCH --job-name=glm46v_Ax_1k
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/glm46v_flash_A%a_1000_%j.out
#SBATCH --error=logs/slurm/glm46v_flash_A%a_1000_%j.err
#
# Controlled GLM A2–A5 @1000 ONLY (no A1 rerun, no 5747).
# Submit one ablation at a time:
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=20260914_glm46v_flash_A2A5_1000,ABLATION=A2 \
#     --job-name=glm46v_A2_1k \
#     --output=logs/slurm/glm46v_flash_A2_1000_%j.out \
#     --error=logs/slurm/glm46v_flash_A2_1000_%j.err \
#     jobs/run_glm_4_6v_flash_Ax_1000.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

ABLATION="${ABLATION:?Set ABLATION to A2, A3, A4, or A5}"
case "${ABLATION}" in
    A2) CONDITION_DIR="A2_overlay_confidence" ;;
    A3) CONDITION_DIR="A3_overlay_confidence_geometry" ;;
    A4) CONDITION_DIR="A4_overlay_crop_confidence" ;;
    A5) CONDITION_DIR="A5_crop_only" ;;
    *)
        echo "ERROR: ABLATION must be A2|A3|A4|A5 (got ${ABLATION})" >&2
        exit 1
        ;;
esac

GLM_VENV="${GLM_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-glm46v}"
module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${GLM_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${GLM_VENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_glm46v_flash_A2A5_1000}"
MODEL="${MODEL:-glm_4_6v_flash}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash}"

PROMPT_INDEX="${PROJECT_DIR}/outputs/verification_ablation_1000/${CONDITION_DIR}/prompt_index.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
EVAL_DIR="${PROJECT_DIR}/outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
DATASET_INDEX="${PROJECT_DIR}/outputs/verification_dataset/index.csv"

mkdir -p logs/slurm "${RESULTS_DIR}" "${EVAL_DIR}"

echo "===== GLM-4.6V-Flash Canonical ${ABLATION}-1000 (native TF5) ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Ablation:      ${ABLATION} (${CONDITION_DIR})"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Evaluation:    ${EVAL_DIR}"
echo "Model path:    ${MODEL_PATH}"
echo "Resume:        ${RESUME}"
echo "GLM_VENV:      ${GLM_VENV}"
echo "Start time:    $(date)"
echo

case "${RESULTS_DIR}" in
    */outputs/verification/qwen/*|*/outputs/verification/qwen2_5_vl/*|*/outputs/evaluation/qwen/*)
        echo "ERROR: refusing Qwen production path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

echo "===== Isolated runtime evidence ====="
which python
python --version
python - <<'PY'
import transformers, torch
print("transformers", transformers.__version__)
print("torch", torch.__version__)
assert transformers.__version__ == "5.17.0", transformers.__version__
from src.lvm.glm_4_6v_flash_verifier import Glm46vFlashVerifier
assert not hasattr(Glm46vFlashVerifier, "_ensure_rope_scaling")
from transformers import Glm4vForConditionalGeneration
print("Glm4vForConditionalGeneration OK; old RoPE patch executed = NO")
PY
case "$(which python)" in
    */envs/wild-palm-glm46v/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-glm46v: $(which python)" >&2
        exit 1
        ;;
esac
nvidia-smi
echo

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi

# Sample-set parity vs A1
python - <<PY
from pathlib import Path
import pandas as pd
a1 = set(pd.read_csv("outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv").sample_id.astype(str))
idx = pd.read_csv("${PROMPT_INDEX}")
assert len(idx) == 1000 and idx.sample_id.nunique() == 1000
assert set(idx.sample_id.astype(str)) == a1
assert set(idx.condition.unique()) == {"${CONDITION_DIR}"}
print("PARITY_PASS: ${ABLATION} IDs identical to A1; N=1000")
PY

VERIFICATION_ARGS=(
    --model glm_4_6v_flash
    --condition "${ABLATION}"
    --limit 1000
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-path "${MODEL_PATH}"
    --model-config "${PROJECT_DIR}/configs/models/glm_4_6v_flash.yaml"
)

if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

echo "===== Inference (${ABLATION}-1000 ONLY; no 5747) ====="
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
echo "${ABLATION}_1000_COMPLETE — hard stop (no 5747)"
