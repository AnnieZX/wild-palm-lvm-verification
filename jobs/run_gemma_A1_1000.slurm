#!/bin/bash
#SBATCH --job-name=gemma_A1_1000
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Gemma 3 12B IT A1 verification (SAMPLE_SIZE=1000).
#
# Protocol-equivalent to LLaVA/Qwen2.5-VL A1: same shared verification dataset,
# same A1 ablation prompts/images under outputs/verification_ablation_1000/,
# same runner / OutputManager / resume / evaluation. Only the model changes.
#
# Submit from the project root:
#   sbatch jobs/run_gemma_A1_1000.slurm
#
# Resume:
#   RESUME=1 EXPERIMENT_ID=<id> sbatch jobs/run_gemma_A1_1000.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
# Default skip rebuild so Gemma reuses the same ablation inputs as Qwen/LLaVA.
SKIP_DATASET_PREP="${SKIP_DATASET_PREP:-1}"
MODEL="${MODEL:-gemma}"
ABLATION="${ABLATION:-A1}"
RESUME="${RESUME:-0}"

ABLATION_INPUTS_DIR="${ABLATION_INPUTS_DIR:-${PROJECT_DIR}/outputs/verification_ablation_${SAMPLE_SIZE}}"
PROMPT_INDEX="${ABLATION_INPUTS_DIR}/A1_overlay_only/prompt_index.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/A1"

mkdir -p logs/slurm
mkdir -p "outputs/verification/${MODEL}/${EXPERIMENT_ID}"
mkdir -p "outputs/evaluation/${MODEL}/${EXPERIMENT_ID}"

echo "===== Gemma A1 Verification Experiment ====="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Submit dir:  ${PROJECT_DIR}"
echo "Model:       ${MODEL}"
echo "Ablation:    ${ABLATION}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Experiment:  ${EXPERIMENT_ID}"
echo "Resume:      ${RESUME}"
echo "Skip prep:   ${SKIP_DATASET_PREP}"
echo "Start time:  $(date)"
echo

echo "Python: $(which python)"
python --version
echo

echo "=== nvidia-smi ==="
nvidia-smi
echo

if [[ "${SKIP_DATASET_PREP}" == "1" ]]; then
    if [[ ! -f "${PROMPT_INDEX}" ]]; then
        echo "ERROR: Shared A1 ablation inputs not found:" >&2
        echo "  ${PROMPT_INDEX}" >&2
        echo "Build once (same as Qwen) with:" >&2
        echo "  SAMPLE_SIZE=${SAMPLE_SIZE} SKIP_DATASET_PREP=0 MODEL=qwen2_5_vl ABLATION=A1 \\" >&2
        echo "    bash scripts/run_qwen_ablation_experiment.sh --ablation A1" >&2
        echo "Or set SKIP_DATASET_PREP=0 for this job to generate them." >&2
        exit 1
    fi
    echo "Using shared ablation inputs: ${ABLATION_INPUTS_DIR}"
    echo
fi

VERIFICATION_ARGS=(
    --model gemma
    --condition A1
    --limit 1000
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
)

if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
echo "End time: $(date)"
echo "Verification: outputs/verification/${MODEL}/${EXPERIMENT_ID}/A1/"
echo "Evaluation:   outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/A1/"
echo "Slurm stdout: ${SLURM_OUTPUT:-logs/slurm/}"
