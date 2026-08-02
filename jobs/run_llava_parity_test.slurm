#!/bin/bash
#SBATCH --job-name=llava_parity
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/llava_parity_%j.out
#SBATCH --error=logs/slurm/llava_parity_%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# LLaVA-OneVision inference parity test (production vs diagnostic paths).
#
# Submit from the project root:
#   sbatch jobs/run_llava_parity_test.slurm
#
# Monitor:
#   tail -f logs/slurm/llava_parity_<jobid>.out
# ---------------------------------------------------------------------------

echo "===== LLaVA Inference Parity Test ====="
echo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/slurm

echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Submit dir:  ${PROJECT_DIR}"
echo "Start time:  $(date)"
echo

echo "Loading environment..."
echo

hostname
pwd
python --version
python -c "import transformers; print(transformers.__version__)"
echo
echo "=== nvidia-smi ==="
nvidia-smi
echo

echo "Running parity test..."
echo

python tests/test_llava_parity.py

echo
echo "Parity test completed."
echo
echo "End time: $(date)"
echo "Slurm log: logs/slurm/llava_parity_${SLURM_JOB_ID:-local}.out"
