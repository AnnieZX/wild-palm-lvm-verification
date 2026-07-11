#!/bin/bash
#SBATCH --job-name=qwen_ablation
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
# Overnight Qwen2.5-VL verification ablation (A1–A5).
#
# Submit from the project root (timestamped logs recommended):
#   bash jobs/submit_qwen_ablation.sh
#
# Or with custom sample size:
#   SAMPLE_SIZE=500 bash jobs/submit_qwen_ablation.sh
#
# Override log paths at submit time via submit_qwen_ablation.sh, or set:
#   #SBATCH --output=logs/slurm/qwen_ablation_%j.out
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

SAMPLE_SIZE="${SAMPLE_SIZE:-300}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SKIP_DATASET_PREP="${SKIP_DATASET_PREP:-0}"
MODEL="${MODEL:-qwen2_5_vl}"

mkdir -p logs/slurm
mkdir -p "outputs/verification/${MODEL}/${EXPERIMENT_ID}"
mkdir -p "outputs/evaluation/${MODEL}/${EXPERIMENT_ID}"

echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Submit dir:  ${PROJECT_DIR}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Experiment:  ${EXPERIMENT_ID}"
echo "Start time:  $(date)"
echo

echo "Python: $(which python)"
python --version
echo

echo "=== nvidia-smi ==="
nvidia-smi
echo

export SAMPLE_SIZE EXPERIMENT_ID BATCH_SIZE SKIP_DATASET_PREP ABLATION RESUME MODEL MODEL_CONFIG

EXPERIMENT_ARGS=()
if [[ -n "${ABLATION:-}" ]]; then
    EXPERIMENT_ARGS+=(--ablation "${ABLATION}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    EXPERIMENT_ARGS+=(--resume)
fi

bash scripts/run_qwen_ablation_experiment.sh "${EXPERIMENT_ARGS[@]}"

echo
echo "End time: $(date)"
echo "Slurm stdout: ${SLURM_OUTPUT:-logs/slurm/}"
