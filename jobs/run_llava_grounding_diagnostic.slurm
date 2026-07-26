#!/bin/bash
#SBATCH --job-name=llava_ground
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
# No --nodelist: let Slurm place the job on any available L40S node in yangGrp.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/llava_grounding_diagnostic_%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# LLaVA-OneVision visual grounding diagnostic (12 samples × prompt variants).
#
# Submit from the project root:
#   sbatch jobs/run_llava_grounding_diagnostic.slurm
#
# Monitor:
#   tail -f logs/slurm/llava_grounding_diagnostic_<jobid>.out
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
echo "Slurm stdout: ${SLURM_OUTPUT:-logs/slurm/llava_grounding_diagnostic_${SLURM_JOB_ID:-local}.out}"
