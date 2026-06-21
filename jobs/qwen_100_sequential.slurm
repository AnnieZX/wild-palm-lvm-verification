#!/bin/bash
#SBATCH --job-name=qwen_100seq
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen_100seq_%j.out
#SBATCH --error=logs/qwen_100seq_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Reproducible 100-palm sequential Qwen2.5-VL experiment.
# Submit from the project root:
#   mkdir -p logs outputs/raw_responses_100_sequential outputs/lvm_inputs_100_sequential
#   sbatch jobs/qwen_100_sequential.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
METADATA_CSV="${PROJECT_DIR}/outputs/lvm_inputs_metadata_100_sequential.csv"
RESULTS_CSV="${PROJECT_DIR}/outputs/qwen_100_sequential_results.csv"
RAW_RESPONSE_DIR="${PROJECT_DIR}/outputs/raw_responses_100_sequential"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Project directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p outputs/raw_responses_100_sequential
mkdir -p outputs/lvm_inputs_100_sequential

echo "Python: $(which python)"
python --version

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== Step 1: Prepare 100 sequential LVM inputs ==="
python scripts/prepare_lvm_inputs_100_sequential.py

if [[ ! -f "${METADATA_CSV}" ]]; then
    echo "ERROR: Metadata file not found after preparation: ${METADATA_CSV}"
    exit 1
fi

ROW_COUNT=$(python - <<'PY'
import pandas as pd
from pathlib import Path
csv_path = Path("outputs/lvm_inputs_metadata_100_sequential.csv")
df = pd.read_csv(csv_path)
print(len(df))
PY
)

if [[ "${ROW_COUNT}" != "100" ]]; then
    echo "ERROR: Expected 100 metadata rows, found ${ROW_COUNT}"
    exit 1
fi

echo "Metadata validation passed: ${ROW_COUNT} rows"

echo
echo "=== Step 2: Run Qwen batch verification ==="
python scripts/run_qwen_batch_100_sequential.py

echo
echo "End time: $(date)"
echo "100-palm sequential experiment complete."
echo "Metadata CSV: ${METADATA_CSV}"
echo "Results CSV: ${RESULTS_CSV}"
echo "Raw responses: ${RAW_RESPONSE_DIR}/"
echo "Stdout log: ${PROJECT_DIR}/logs/qwen_100seq_${SLURM_JOB_ID}.out"
echo "Stderr log: ${PROJECT_DIR}/logs/qwen_100seq_${SLURM_JOB_ID}.err"
