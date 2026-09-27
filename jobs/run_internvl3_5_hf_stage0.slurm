#!/bin/bash
#SBATCH --job-name=ivl35hf_s0
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/internvl3_5_hf_stage0_%j.out
#SBATCH --error=logs/slurm/internvl3_5_hf_stage0_%j.err
# Prefer group L40S:
#   sbatch --partition=yangGrp --nodelist=lovelace --gres=gpu:L40S:1 \
#     jobs/run_internvl3_5_hf_stage0.slurm
# H200 alternative:
#   sbatch --partition=gpu_small --gres=gpu:H200_141:1 \
#     jobs/run_internvl3_5_hf_stage0.slurm

set -euo pipefail

# ---------------------------------------------------------------------------
# InternVL3.5-8B-HF Stage 0 qualification (10 diagnostic samples).
#
# Uses fixed prompt_index under outputs/diagnostics/model_qualification/stage0_A1_10/
# referencing canonical A1-1000 prompts/images (no regeneration).
#
# Submit from project root:
#   EXPERIMENT_ID=YYYYMMDD_internvl3_5_hf_qual sbatch \
#     --partition=yangGrp --nodelist=lovelace --gres=gpu:L40S:1 \
#     jobs/run_internvl3_5_hf_stage0.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_internvl3_5_hf_qual}"
MODEL="${MODEL:-internvl3_5_hf}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"

PROMPT_INDEX="${PROMPT_INDEX:-${PROJECT_DIR}/outputs/diagnostics/model_qualification/stage0_A1_10/prompt_index.csv}"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/stage0"

mkdir -p logs/slurm
mkdir -p "outputs/verification/${MODEL}/${EXPERIMENT_ID}/stage0"

echo "===== InternVL3.5-HF Stage 0 Qualification ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "Submit dir:    ${PROJECT_DIR}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Resume:        ${RESUME}"
echo "Start time:    $(date)"
echo

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
echo "Python: $(which python)"
python --version
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from transformers import AutoModelForImageTextToText, AutoProcessor, InternVLForConditionalGeneration; print('InternVL HF classes OK')"
echo
echo "=== nvidia-smi ==="
nvidia-smi
echo

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: Stage 0 prompt index not found: ${PROMPT_INDEX}" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/InternVL3_5-8B-HF}"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Model checkpoint not found: ${MODEL_PATH}" >&2
    exit 1
fi
echo "Model path: ${MODEL_PATH}"
if [[ -f "${MODEL_PATH}/DOWNLOAD_META.txt" ]]; then
    cat "${MODEL_PATH}/DOWNLOAD_META.txt"
fi
if [[ -f "${MODEL_PATH}/config.json" ]]; then
    python - <<PY
import json
cfg = json.load(open("${MODEL_PATH}/config.json"))
print("architectures:", cfg.get("architectures"))
print("model_type:", cfg.get("model_type"))
print("torch_dtype:", cfg.get("torch_dtype"))
print("transformers_version:", cfg.get("transformers_version"))
print("auto_map:", cfg.get("auto_map"))
PY
fi
echo

VERIFICATION_ARGS=(
    --model internvl3_5_hf
    --condition stage0
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-path "${MODEL_PATH}"
)

if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
echo "End time: $(date)"
echo "Verification: ${RESULTS_DIR}/"
echo "Wrote experiment id: ${EXPERIMENT_ID}"
