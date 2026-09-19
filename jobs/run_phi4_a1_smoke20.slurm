#!/bin/bash
#SBATCH --job-name=phi4_smoke20
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/phi4_smoke20_%j.out
#SBATCH --error=logs/slurm/phi4_smoke20_%j.err
#
# Gate 1: Phi-4 Multimodal Instruct A1 smoke N=20 (first 20 of canonical A1-1000).
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=YYYYMMDD_HHMM_phi4_smoke20 \
#     jobs/run_phi4_a1_smoke20.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

PHI_VENV="${PHI_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-phi4}"
module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${PHI_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${PHI_VENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_phi4_smoke20}"
MODEL="${MODEL:-phi4_multimodal}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Phi-4-multimodal-instruct}"
SAMPLE_SIZE=20

PROMPT_INDEX="${PROJECT_DIR}/outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/A1"
EVAL_DIR="${PROJECT_DIR}/outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/A1"
DATASET_INDEX="${PROJECT_DIR}/outputs/verification_dataset/index.csv"

mkdir -p logs/slurm "${RESULTS_DIR}" "${EVAL_DIR}"

echo "===== Phi-4 Multimodal A1 Smoke20 ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Evaluation:    ${EVAL_DIR}"
echo "Model path:    ${MODEL_PATH}"
echo "Resume:        ${RESUME}"
echo "PHI_VENV:      ${PHI_VENV}"
echo "Start time:    $(date)"
echo

case "${RESULTS_DIR}" in
    */outputs/verification/qwen/*|*/outputs/verification/qwen2_5_vl/*|*/outputs/evaluation/qwen/*|\
    */outputs/verification/glm_4_6v_flash/*|*/outputs/evaluation/glm_4_6v_flash/*)
        echo "ERROR: refusing Qwen/GLM production path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

case "$(which python)" in
    */envs/wild-palm-phi4/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-phi4: $(which python)" >&2
        exit 1
        ;;
esac

which python
python --version
python -c 'import transformers,torch,peft; print("transformers",transformers.__version__); print("torch",torch.__version__); print("peft",peft.__version__)'
nvidia-smi
echo

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi

python - <<'PY'
import pandas as pd
idx = pd.read_csv("outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv")
assert len(idx) == 1000
first20 = idx.head(20)["sample_id"].tolist()
assert first20[0] == "sample_000001"
assert len(first20) == 20
print("SMOKE20_IDS:", ",".join(first20))
PY

VERIFICATION_ARGS=(
    --model phi4_multimodal
    --condition A1
    --limit "${SAMPLE_SIZE}"
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-path "${MODEL_PATH}"
    --model-config "${PROJECT_DIR}/configs/models/phi4_multimodal.yaml"
)

if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

echo "===== Inference (A1 smoke N=20 ONLY) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
echo "===== Canonical evaluation ====="
python -u scripts/evaluate_verification_against_groundtruth.py \
    --results-dir "${RESULTS_DIR}" \
    --index-csv "${DATASET_INDEX}" \
    --output-dir "${EVAL_DIR}" \
    --condition-code A1

python -u scripts/compute_verification_metrics.py \
    --evaluation-dir "${EVAL_DIR}"

echo
python - <<PY
import json
from collections import Counter
from pathlib import Path
root = Path("${RESULTS_DIR}")
files = sorted(root.glob("sample_*.json"))
c = Counter()
parse = infer = ok = 0
for f in files:
    data = json.loads(f.read_text())
    st = (data.get("status") or "").lower()
    if st == "parse_error":
        parse += 1
    elif st == "inference_error":
        infer += 1
    else:
        ok += 1
    c[data.get("decision") or "(empty)"] += 1
print("N=", len(files))
print("status ok/parse/infer =", ok, parse, infer)
print("decisions=", dict(c))
print("SMOKE20_SUMMARY_DONE")
PY

if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "Evaluation: ${EVAL_DIR}"
echo "A1_SMOKE20_COMPLETE — hard stop (no A1-1000 / 5747)"
