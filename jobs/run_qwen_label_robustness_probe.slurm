#!/bin/bash
#SBATCH --job-name=qwen_label_rob
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/qwen_label_robustness_%j.out
#SBATCH --error=logs/slurm/qwen_label_robustness_%j.err
#
# DIAGNOSTIC ONLY — 20 samples × 4 conditions = 80 inference cases.
# Does NOT touch job 8318954 / A5 production outputs.
#
# Submit example (another L40S on lovelace; queues if needed):
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 --nodelist=lovelace \
#     jobs/run_qwen_label_robustness_probe.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

OUT_ROOT="${PROJECT_DIR}/outputs/diagnostics/model_qualification/qwen_label_robustness_20"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct}"
PROD_A5_JOB="${PROD_A5_JOB:-8318954}"

mkdir -p logs/slurm

echo "===== Qwen2.5-VL Label-Robustness Diagnostic ====="
echo "DIAGNOSTIC ONLY — isolated under outputs/diagnostics/"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Out root:     ${OUT_ROOT}"
echo "Model:        ${MODEL_PATH}"
echo "Start:        $(date)"
echo

echo "--- Production A5 job status (must not interfere) ---"
squeue -j "${PROD_A5_JOB}" -o '%.18i %.12P %.20j %.8u %.2t %.10M %R %b' || true
echo

python --version
python -c "import transformers; print('transformers', transformers.__version__)"
nvidia-smi
echo

if [[ ! -f "${OUT_ROOT}/prompt_index.csv" ]]; then
    echo "ERROR: missing prepared prompt_index at ${OUT_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${OUT_ROOT}/prompt_parity_audit/VERDICT.json" ]]; then
    echo "ERROR: missing prompt-parity VERDICT; prepare/audit first." >&2
    exit 1
fi
python - <<'PY'
import json, sys
from pathlib import Path
v = json.loads(Path("outputs/diagnostics/model_qualification/qwen_label_robustness_20/prompt_parity_audit/VERDICT.json").read_text())
if not v.get("prompt_parity_ok"):
    print("PROMPT PARITY FAILED — refusing to run.", file=sys.stderr)
    sys.exit(2)
print("Prompt parity OK")
PY

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing model ${MODEL_PATH}" >&2
    exit 1
fi

# Guard: never write outside diagnostics
case "${OUT_ROOT}" in
    */outputs/diagnostics/*) ;;
    *)
        echo "ERROR: refusing non-diagnostics path: ${OUT_ROOT}" >&2
        exit 1
        ;;
esac

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "===== Inference (80 cases) ====="
python -u scripts/diagnostics/run_qwen_label_robustness_probe.py \
    --out-root "${OUT_ROOT}" \
    --model-path "${MODEL_PATH}"

echo
echo "===== Evaluation ====="
python -u scripts/diagnostics/evaluate_qwen_label_robustness_probe.py \
    --out-root "${OUT_ROOT}"

echo
echo "End: $(date)"
echo "Results: ${OUT_ROOT}"
