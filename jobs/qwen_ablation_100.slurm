#!/bin/bash
#SBATCH --job-name=qwen_ablation100
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/qwen_ablation100_%j.out
#SBATCH --error=logs/qwen_ablation100_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 100-palm ablation study: visual overlays x prompt variants.
# Submit from the project root:
#   mkdir -p logs outputs/ablation_inputs_100 outputs/ablation_results_100 \
#            outputs/ablation_raw_responses_100
#   sbatch jobs/qwen_ablation_100.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
METADATA_CSV="${PROJECT_DIR}/outputs/ablation_metadata_100.csv"
COMBINED_CSV="${PROJECT_DIR}/outputs/ablation_results_100_combined.csv"
SUMMARY_CSV="${PROJECT_DIR}/outputs/ablation_summary_100.csv"
ENDPOINT_SUMMARY_CSV="${PROJECT_DIR}/outputs/ablation_endpoint_summary_100.csv"
ANCHORING_SUMMARY_CSV="${PROJECT_DIR}/outputs/ablation_anchoring_summary_100.csv"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Project directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p outputs/ablation_inputs_100
mkdir -p outputs/ablation_results_100
mkdir -p outputs/ablation_raw_responses_100

echo "Python: $(which python)"
python --version

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== Step 1: Prepare ablation inputs ==="
python scripts/prepare_ablation_inputs_100.py

if [[ ! -f "${METADATA_CSV}" ]]; then
    echo "ERROR: Metadata file not found after preparation: ${METADATA_CSV}"
    exit 1
fi

ROW_COUNT=$(python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv(Path("outputs/ablation_metadata_100.csv"))
print(len(df))
PY
)

if [[ "${ROW_COUNT}" != "100" ]]; then
    echo "ERROR: Expected 100 metadata rows, found ${ROW_COUNT}"
    exit 1
fi

echo "Metadata validation passed: ${ROW_COUNT} rows"

echo
echo "=== Step 2: Run ablation matrix ==="
python scripts/run_qwen_ablation_100.py

echo
echo "=== Step 3: Analyze ablation results ==="
python scripts/analyze_ablation_100.py

COMBINED_ROWS=$(python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path("outputs/ablation_results_100_combined.csv")
if not path.exists():
    print(0)
else:
    print(len(pd.read_csv(path)))
PY
)

if [[ "${COMBINED_ROWS}" != "1000" ]]; then
    echo "WARNING: Expected 1000 combined rows, found ${COMBINED_ROWS}"
fi

echo
echo "End time: $(date)"
echo "100-palm ablation experiment complete."
echo "Final outputs:"
echo "  ${METADATA_CSV}"
echo "  ${COMBINED_CSV}"
echo "  ${SUMMARY_CSV}"
echo "  ${ENDPOINT_SUMMARY_CSV}"
echo "  ${ANCHORING_SUMMARY_CSV}"
echo "Stdout log: ${PROJECT_DIR}/logs/qwen_ablation100_${SLURM_JOB_ID}.out"
echo "Stderr log: ${PROJECT_DIR}/logs/qwen_ablation100_${SLURM_JOB_ID}.err"
