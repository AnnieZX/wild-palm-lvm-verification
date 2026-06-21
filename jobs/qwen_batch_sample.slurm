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

# ---------------------------------------------------------------------------
# Batch Qwen2.5-VL verification for sample palms (15 instances).
# Submit from the project root:
#   mkdir -p logs outputs/raw_responses
#   sbatch jobs/qwen_batch_sample.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
METADATA_CSV="${PROJECT_DIR}/outputs/lvm_inputs_metadata.csv"
RESULTS_CSV="${PROJECT_DIR}/outputs/qwen_sample_results.csv"
RAW_RESPONSE_DIR="${PROJECT_DIR}/outputs/raw_responses"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Project directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p outputs/raw_responses

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate palm-lvm

echo "Python: $(which python)"
python --version

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== Cluster environment check ==="
python scripts/check_cluster_environment.py

if [[ ! -f "${METADATA_CSV}" ]]; then
    echo "ERROR: Metadata file not found: ${METADATA_CSV}"
    echo "Run prepare_lvm_inputs.py before submitting this job."
    exit 1
fi

echo
echo "=== Running batch Qwen verification ==="
python scripts/run_qwen_batch_sample.py

echo
echo "End time: $(date)"
echo "Batch job complete."
echo "Results CSV: ${RESULTS_CSV}"
echo "Raw responses: ${RAW_RESPONSE_DIR}/"
echo "Stdout log: ${PROJECT_DIR}/logs/qwen_batch_sample_${SLURM_JOB_ID}.out"
echo "Stderr log: ${PROJECT_DIR}/logs/qwen_batch_sample_${SLURM_JOB_ID}.err"
