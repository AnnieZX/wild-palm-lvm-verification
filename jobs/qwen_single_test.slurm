#!/bin/bash
#SBATCH --job-name=qwen_single_test
#SBATCH --partition=yangGrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen_single_test-%j.out
#SBATCH --error=logs/qwen_single_test-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Run a single-image Qwen verification test on the DEAC cluster.
# Submit from the project root:
#   mkdir -p logs
#   sbatch jobs/qwen_single_test.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Project directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate palm-lvm

echo "Python: $(which python)"
python --version

echo "Running cluster environment check..."
python scripts/check_cluster_environment.py

echo "Running single-image Qwen test..."
python scripts/run_qwen_single.py

echo "End time: $(date)"
echo "Logs saved under: ${PROJECT_DIR}/logs/"
