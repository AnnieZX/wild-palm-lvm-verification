#!/bin/bash
#SBATCH --job-name=phi4_Ax_5747
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/phi4_A%a_5747_%j.out
#SBATCH --error=logs/slurm/phi4_A%a_5747_%j.err
#
# Phi-4 Multimodal Instruct canonical A1–A5 @5747 (full verification set).
# PREPARED but submit only AFTER A2 @1000 screening passes.
# Submit one ablation at a time (preferred: L40S):
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=YYYYMMDD_phi4_A1A5_5747,ABLATION=A1 \
#     --job-name=phi4_A1_5747 \
#     --output=logs/slurm/phi4_A1_5747_%j.out \
#     --error=logs/slurm/phi4_A1_5747_%j.err \
#     jobs/run_phi4_Ax_5747.slurm

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

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_phi4_A1A5_5747}"
MODEL="${MODEL:-phi4_multimodal}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Phi-4-multimodal-instruct}"
SAMPLE_SIZE=5747

PROMPT_INDEX="${PROJECT_DIR}/outputs/verification_ablation_5747/${CONDITION_DIR}/prompt_index.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
EVAL_DIR="${PROJECT_DIR}/outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/${ABLATION}"
DATASET_INDEX="${PROJECT_DIR}/outputs/verification_dataset/index.csv"

mkdir -p logs/slurm "${RESULTS_DIR}" "${EVAL_DIR}"

echo "===== Phi-4 Multimodal Canonical ${ABLATION}-5747 ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Ablation:      ${ABLATION} (${CONDITION_DIR})"
echo "Sample size:   ${SAMPLE_SIZE}"
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
    */outputs/verification/glm_4_6v_flash/*|*/outputs/evaluation/glm_4_6v_flash/*|\
    */20260919_1524_phi4_A1A5_1000/*)
        echo "ERROR: refusing protected path: ${RESULTS_DIR}" >&2
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

# Fail fast on GPUs with active uncorrectable ECC.
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,ecc.errors.uncorrected.volatile.total",
     "--format=csv,noheader,nounits"],
    text=True,
)
bad = []
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    idx, ecc = parts[0], parts[1]
    try:
        ecc_n = int(ecc)
    except ValueError:
        ecc_n = -1
    print(f"GPU {idx} volatile_uncorrected_ecc_total={ecc}")
    if ecc_n > 0:
        bad.append((idx, ecc_n))
if bad:
    print("ERROR: refusing GPU(s) with uncorrected ECC:", bad, file=sys.stderr)
    sys.exit(2)
print("ECC_GUARD_PASS")
PY

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi

# Full-set integrity + parity vs A1 @5747
python - <<PY
import pandas as pd
a1 = set(pd.read_csv("outputs/verification_ablation_5747/A1_overlay_only/prompt_index.csv").sample_id.astype(str))
idx = pd.read_csv("${PROMPT_INDEX}")
assert len(idx) == 5747 and idx.sample_id.nunique() == 5747
assert set(idx.sample_id.astype(str)) == a1
assert set(idx.condition.unique()) == {"${CONDITION_DIR}"}
print("PARITY_PASS: ${ABLATION} IDs identical to A1; N=5747")
PY

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

echo "===== Inference (${ABLATION}-5747) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

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
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "Evaluation: ${EVAL_DIR}"
echo "${ABLATION}_5747_COMPLETE"
