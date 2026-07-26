#!/bin/bash
# ---------------------------------------------------------------------------
# TEMPORARY diagnostic script — public GPU partition test only.
#
# Do NOT use this in place of jobs/run_llava_grounding_diagnostic.slurm.
# Production runs should stay on yangGrp with --gres=gpu:L40S:1.
#
# Purpose: verify that the current environment can run the LLaVA grounding
# diagnostic on another GPU partition before relying on yangGrp/L40S.
#
# Partition is configurable at submit time (default below):
#   sbatch --partition=<public_gpu_partition> jobs/run_llava_grounding_gpu_test.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=llava_ground_gpu
#SBATCH --partition=gpu
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/llava_grounding_gpu_test_%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# LLaVA-OneVision visual grounding diagnostic (12 samples × prompt variants).
#
# Submit from the project root:
#   sbatch --partition=<public_gpu_partition> jobs/run_llava_grounding_gpu_test.slurm
#
# Monitor:
#   tail -f logs/slurm/llava_grounding_gpu_test_<jobid>.out
# ---------------------------------------------------------------------------

echo "===== LLaVA Grounding Diagnostic ====="
echo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/slurm outputs/debug/llava_grounding

echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Submit dir:  ${PROJECT_DIR}"
echo "Start time:  $(date)"
echo

hostname
pwd
python --version
python -c "import transformers; print(transformers.__version__)"
echo
echo "=== nvidia-smi ==="
nvidia-smi
echo

EVAL_CSV=""
if [[ -f "outputs/evaluation/A1_evaluation.csv" ]]; then
  EVAL_CSV="--evaluation-csv outputs/evaluation/A1_evaluation.csv"
elif [[ -f "outputs/evaluation/llava/A1_evaluation.csv" ]]; then
  EVAL_CSV="--evaluation-csv outputs/evaluation/llava/A1_evaluation.csv"
fi

python scripts/debug_llava_grounding.py \
  --output-dir outputs/debug/llava_grounding \
  --sample-size 1000 \
  --seed 42 \
  --max-new-tokens 512 \
  ${EVAL_CSV}

echo
echo "Diagnostic completed."
echo "Outputs: outputs/debug/llava_grounding/"
echo "End time: $(date)"
echo "Slurm stdout: ${SLURM_OUTPUT:-logs/slurm/llava_grounding_gpu_test_${SLURM_JOB_ID:-local}.out}"
