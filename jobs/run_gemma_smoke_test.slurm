#!/bin/bash
#SBATCH --job-name=gemma_smoke
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/gemma_smoke_%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# Gemma 3 12B IT infrastructure smoke test (load + single-image inference).
#
# Submit from the project root:
#   sbatch jobs/run_gemma_smoke_test.slurm
#
# Monitor:
#   tail -f logs/slurm/gemma_smoke_<jobid>.out
# ---------------------------------------------------------------------------

echo "===== Gemma 3 Smoke Test ====="
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

echo "Running smoke test..."
echo

python scripts/smoke_tests/test_gemma3.py \
    --image data/samples/images/100_0003_0001_1.png \
    --model-path /deac/csc/yangGrp/luoz23/models/gemma-3-12b-it

echo
echo "Smoke test completed."
echo
echo "End time: $(date)"
echo "Slurm stdout: ${SLURM_OUTPUT:-logs/slurm/gemma_smoke_${SLURM_JOB_ID:-local}.out}"
