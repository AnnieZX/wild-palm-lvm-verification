#!/bin/bash
#SBATCH --job-name=glm46v_s0_native
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/glm46v_flash_stage0_%j.out
#SBATCH --error=logs/slurm/glm46v_flash_stage0_%j.err
#
# Stage-0 ONLY (N=10). Native Transformers 5.x isolated env. Submit:
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=20260913_glm46v_flash_stage0_native \
#     jobs/run_glm_4_6v_flash_stage0.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

GLM_VENV="${GLM_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-glm46v}"
# Prefer module Python that built the venv; then activate venv.
module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${GLM_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${GLM_VENV}/bin:${PATH}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_glm46v_flash_stage0_native}"
MODEL="${MODEL:-glm_4_6v_flash}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash}"

PROMPT_INDEX="${PROMPT_INDEX:-${PROJECT_DIR}/outputs/diagnostics/model_qualification/stage0_A1_10/prompt_index.csv}"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/stage0"

mkdir -p logs/slurm "${RESULTS_DIR}"

echo "===== GLM-4.6V-Flash Stage 0 Qualification (N=10, native TF5) ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Model path:    ${MODEL_PATH}"
echo "Resume:        ${RESUME}"
echo "GLM_VENV:      ${GLM_VENV}"
echo "Start time:    $(date)"
echo

# Hard safety: never write into Qwen production trees
case "${RESULTS_DIR}" in
    */outputs/verification/qwen/*|*/outputs/verification/qwen2_5_vl/*|*/outputs/evaluation/qwen/*)
        echo "ERROR: refusing Qwen production path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

# Required evidence: Slurm is using the isolated runtime
echo "===== Isolated runtime evidence ====="
which python
python --version
python -c "import transformers; print(transformers.__version__)"
case "$(which python)" in
    */envs/wild-palm-glm46v/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-glm46v: $(which python)" >&2
        exit 1
        ;;
esac
python -c "from transformers import Glm4vForConditionalGeneration, AutoProcessor; print('Glm4v OK')"
nvidia-smi
echo

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing Stage-0 prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -f "${MODEL_PATH}/DOWNLOAD_META.txt" ]]; then
    cat "${MODEL_PATH}/DOWNLOAD_META.txt"
fi

# Print Stage-0 sample IDs + GT before inference
python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path("outputs/diagnostics/model_qualification/stage0_A1_10")
man = pd.read_csv(root / "manifest.csv")
print("===== Stage-0 sample set (exact InternVL stage0_A1_10) =====")
print(man[["sample_id", "gt_label"]].to_string(index=False))
print("n=", len(man))
assert len(man) == 10
PY

# Prompt parity vs canonical Qwen A1 (semantic content)
python - <<'PY'
from pathlib import Path
import pandas as pd
import difflib

root = Path("outputs/diagnostics/model_qualification/stage0_A1_10")
idx = pd.read_csv(root / "prompt_index.csv")
# Resolve relative to prompt_index parent
base = root
fail = False
for _, row in idx.iterrows():
    sid = row["sample_id"]
    glm_prompt_path = (base / row["prompt_path"]).resolve()
    # Canonical A1 prompt is the same file referenced by stage0 index
    can_path = Path("outputs/verification_ablation_1000/A1_overlay_only/prompts") / f"{sid}.txt"
    glm_text = glm_prompt_path.read_text(encoding="utf-8")
    can_text = can_path.read_text(encoding="utf-8")
    if glm_text != can_text:
        fail = True
        print(f"PARITY FAIL {sid}")
        print("".join(difflib.unified_diff(can_text.splitlines(True), glm_text.splitlines(True), "canonical", "stage0", n=2))[:500])
        break
    # Spot-check decision labels and A1 constraints
    for token in ["Reliable", "Uncertain", "Unreliable"]:
        assert token in glm_text
    assert "YOLO confidence" not in glm_text or "Do not use YOLO confidence" in glm_text
    assert "A1_overlay_only" in glm_text
if fail:
    raise SystemExit("PROMPT_PARITY_FAIL")
print("PROMPT_PARITY_PASS: stage0 prompts byte-identical to canonical A1_overlay_only")
print("Semantic decisions: Reliable / Uncertain / Unreliable (unchanged order)")
print("No confidence/geometry metadata; overlay-only A1 visual")
print("Model-specific wrapping is chat-template only (applied at inference)")
PY

VERIFICATION_ARGS=(
    --model glm_4_6v_flash
    --condition stage0
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-path "${MODEL_PATH}"
    --model-config "${PROJECT_DIR}/configs/models/glm_4_6v_flash.yaml"
)

if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

echo "===== Inference ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
