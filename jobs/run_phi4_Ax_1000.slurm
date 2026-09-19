#!/bin/bash
#SBATCH --job-name=phi4_Ax_1k
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/phi4_A%a_1000_%j.out
#SBATCH --error=logs/slurm/phi4_A%a_1000_%j.err
#
# Gate 2: Phi-4 Multimodal Instruct frozen A1–A5 @1000 ONLY (no 5747).
# Submit one ablation at a time:
#   sbatch --partition=gpu_small --gres=gpu:V100_32:1 \
#     --export=ALL,EXPERIMENT_ID=YYYYMMDD_HHMM_phi4_A1A5_1000,ABLATION=A1 \
#     --job-name=phi4_A1_1k \
#     --output=logs/slurm/phi4_A1_1000_%j.out \
#     --error=logs/slurm/phi4_A1_1000_%j.err \
#     jobs/run_phi4_Ax_1000.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

ABLATION="${ABLATION:?Set ABLATION to A1, A2, A3, A4, or A5}"
case "${ABLATION}" in
    A1) CONDITION_DIR="A1_overlay_only" ;;
    A2) CONDITION_DIR="A2_overlay_confidence" ;;
    A3) CONDITION_DIR="A3_overlay_confidence_geometry" ;;
    A4) CONDITION_DIR="A4_overlay_crop_confidence" ;;
    A5) CONDITION_DIR="A5_crop_only" ;;
    *)
        echo "ERROR: ABLATION must be A1|A2|A3|A4|A5 (got ${ABLATION})" >&2
        exit 1
        ;;
esac

PHI_VENV="${PHI_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-phi4}"
module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${PHI_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${PHI_VENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_phi4_A1A5_1000}"
MODEL="${MODEL:-phi4_multimodal}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Phi-4-multimodal-instruct}"
SAMPLE_SIZE=1000

PROMPT_INDEX="${PROJECT_DIR}/outputs/verification_ablation_1000/${CONDITION_DIR}/prompt_index.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
EVAL_DIR="${PROJECT_DIR}/outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
DATASET_INDEX="${PROJECT_DIR}/outputs/verification_dataset/index.csv"

mkdir -p logs/slurm "${RESULTS_DIR}" "${EVAL_DIR}"

echo "===== Phi-4 Multimodal Canonical ${ABLATION}-1000 ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Ablation:      ${ABLATION} (${CONDITION_DIR})"
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

# Canonical 1000-sample parity vs A1 (no resampling)
python - <<PY
from pathlib import Path
import pandas as pd
a1 = set(pd.read_csv("outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv").sample_id.astype(str))
idx = pd.read_csv("${PROMPT_INDEX}")
assert len(idx) == 1000 and idx.sample_id.nunique() == 1000
assert set(idx.sample_id.astype(str)) == a1
assert set(idx.condition.unique()) == {"${CONDITION_DIR}"}
print("PARITY_PASS: ${ABLATION} IDs identical to A1; N=1000")
print("FIRST5:", ",".join(idx.head(5)["sample_id"].astype(str).tolist()))
PY

# Lightweight peak-VRAM sampler (job-local; does not touch protocol code)
VRAM_LOG="${RESULTS_DIR}/peak_vram_monitor.csv"
(
  echo "timestamp,memory_used_mib"
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "$(date -Iseconds),${used:-NA}"
    sleep 30
  done
) >"${VRAM_LOG}" &
VRAM_PID=$!
trap 'kill ${VRAM_PID} 2>/dev/null || true' EXIT

VERIFICATION_ARGS=(
    --model phi4_multimodal
    --condition "${ABLATION}"
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

echo "===== Inference (${ABLATION}-1000 ONLY; no 5747) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

kill "${VRAM_PID}" 2>/dev/null || true
trap - EXIT

echo
echo "===== Peak VRAM (monitor) ====="
python - <<PY
from pathlib import Path
p = Path("${VRAM_LOG}")
vals = []
if p.exists():
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].isdigit():
            vals.append(int(parts[1]))
if vals:
    print(f"peak_vram_used_mib: {max(vals)}")
    print(f"peak_vram_used_gb: {max(vals)/1024:.3f}")
else:
    print("peak_vram_used_mib: NA")
PY
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true

echo
echo "===== Canonical evaluation ====="
python -u scripts/evaluate_verification_against_groundtruth.py \
    --results-dir "${RESULTS_DIR}" \
    --index-csv "${DATASET_INDEX}" \
    --output-dir "${EVAL_DIR}" \
    --condition-code "${ABLATION}"

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
    if data.get("parse_error"):
        parse += 1
    elif data.get("inference_error"):
        infer += 1
    else:
        ok += 1
    c[data.get("decision") or "(empty)"] += 1
print("N=", len(files))
print("status ok/parse/infer =", ok, parse, infer)
print("decisions=", dict(c))
print("${ABLATION}_1000_SUMMARY_DONE")
PY

echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "Evaluation: ${EVAL_DIR}"
echo "${ABLATION}_1000_COMPLETE — hard stop (no 5747)"
