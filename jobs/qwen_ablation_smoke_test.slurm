#!/bin/bash
#SBATCH --job-name=qwen_ablation_smoke
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/qwen_ablation_smoke_%j.out
#SBATCH --error=logs/qwen_ablation_smoke_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Smoke test for the 100-palm ablation pipeline (2 palms x 10 conditions).
# Requires ablation inputs to be prepared first:
#   python scripts/prepare_ablation_inputs_100.py
#
# Submit from the project root:
#   mkdir -p logs outputs/smoke_test_results outputs/smoke_test_raw_responses
#   sbatch jobs/qwen_ablation_smoke_test.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
METADATA_CSV="${PROJECT_DIR}/outputs/ablation_metadata_100.csv"
COMBINED_CSV="${PROJECT_DIR}/outputs/smoke_test_combined.csv"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Project directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p outputs/smoke_test_results
mkdir -p outputs/smoke_test_raw_responses

echo "Python: $(which python)"
python --version

echo
echo "=== nvidia-smi ==="
nvidia-smi

if [[ ! -f "${METADATA_CSV}" ]]; then
    echo "ERROR: Metadata file not found: ${METADATA_CSV}"
    echo "Run prepare_ablation_inputs_100.py first."
    exit 1
fi

ROW_COUNT=$(python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv(Path("outputs/ablation_metadata_100.csv"))
print(len(df))
PY
)

if [[ "${ROW_COUNT}" -lt 2 ]]; then
    echo "ERROR: Need at least 2 metadata rows for smoke test, found ${ROW_COUNT}"
    exit 1
fi

echo "Metadata available: ${ROW_COUNT} rows (smoke test uses first 2)"

echo
echo "=== Running ablation smoke test ==="
python scripts/run_qwen_ablation_smoke_test.py

COMBINED_ROWS=$(python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path("outputs/smoke_test_combined.csv")
if not path.exists():
    print(0)
else:
    print(len(pd.read_csv(path)))
PY
)

echo
echo "End time: $(date)"
echo "Smoke test job complete."
echo "Combined rows: ${COMBINED_ROWS} (expected 20)"
echo "Combined CSV: ${COMBINED_CSV}"
echo "Results dir: ${PROJECT_DIR}/outputs/smoke_test_results/"
echo "Raw responses: ${PROJECT_DIR}/outputs/smoke_test_raw_responses/"
echo "Stdout log: ${PROJECT_DIR}/logs/qwen_ablation_smoke_${SLURM_JOB_ID}.out"
echo "Stderr log: ${PROJECT_DIR}/logs/qwen_ablation_smoke_${SLURM_JOB_ID}.err"

if [[ "${COMBINED_ROWS}" != "20" ]]; then
    echo "WARNING: Expected 20 combined rows, found ${COMBINED_ROWS}"
    exit 1
fi
