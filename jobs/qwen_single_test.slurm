#!/bin/bash
#SBATCH --job-name=qwen_batch_sample
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qwen_batch_sample_%j.out
#SBATCH --error=logs/qwen_batch_sample_%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"

mkdir -p logs
mkdir -p outputs/raw_responses

echo "Python: $(which python)"
python --version

echo "Checking GPU:"
nvidia-smi || true

python scripts/check_cluster_environment.py

echo "Running Qwen batch sample..."
python scripts/run_qwen_batch_sample.py

echo "End time: $(date)"