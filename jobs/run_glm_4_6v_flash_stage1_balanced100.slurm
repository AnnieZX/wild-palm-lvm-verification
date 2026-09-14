#!/bin/bash
#SBATCH --job-name=glm46v_s1_b100
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/glm46v_flash_stage1_%j.out
#SBATCH --error=logs/slurm/glm46v_flash_stage1_%j.err
#
# Stage-1 ONLY (balanced A1-100). Native Transformers 5.x isolated env.
# Submit:
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=20260913_glm46v_flash_stage1_balanced100 \
#     jobs/run_glm_4_6v_flash_stage1_balanced100.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

GLM_VENV="${GLM_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-glm46v}"
module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${GLM_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${GLM_VENV}/bin:${PATH}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_glm46v_flash_stage1_balanced100}"
MODEL="${MODEL:-glm_4_6v_flash}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash}"

PROMPT_INDEX="${PROMPT_INDEX:-${PROJECT_DIR}/outputs/diagnostics/model_qualification/balanced_A1_100/prompt_index.csv}"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/balanced100"

mkdir -p logs/slurm "${RESULTS_DIR}"

echo "===== GLM-4.6V-Flash Stage 1 Balanced-100 (native TF5) ====="
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

case "${RESULTS_DIR}" in
    */outputs/verification/qwen/*|*/outputs/verification/qwen2_5_vl/*|*/outputs/evaluation/qwen/*)
        echo "ERROR: refusing Qwen production path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

echo "===== Isolated runtime evidence ====="
which python
python --version
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
case "$(which python)" in
    */envs/wild-palm-glm46v/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-glm46v: $(which python)" >&2
        exit 1
        ;;
esac
python - <<'PY'
import sys
import transformers
from transformers import Glm4vForConditionalGeneration, AutoProcessor
assert sys.version_info[:2] == (3, 11), sys.version
assert transformers.__version__ == "5.17.0", transformers.__version__
from src.lvm.glm_4_6v_flash_verifier import Glm46vFlashVerifier
assert not hasattr(Glm46vFlashVerifier, "_ensure_rope_scaling")
print("Glm4v OK; old RoPE patch executed = NO")
PY
nvidia-smi
echo

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing balanced100 prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -f "${MODEL_PATH}/DOWNLOAD_META.txt" ]]; then
    cat "${MODEL_PATH}/DOWNLOAD_META.txt"
fi

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs/diagnostics/model_qualification/balanced_A1_100")
man = pd.read_csv(root / "manifest.csv")
idx = pd.read_csv(root / "prompt_index.csv")
print("===== Stage-1 sample set (shared balanced_A1_100) =====")
print("n=", len(man))
print(man["gt_label"].value_counts().to_string())
assert len(man) == 100
assert man["sample_id"].nunique() == 100
assert set(man["sample_id"]) == set(idx["sample_id"])
assert int((man["gt_label"] == "positive").sum()) == 50
assert int((man["gt_label"] == "negative").sum()) == 50
print("GT: greedy matched_gt (IoU>=0.5 one-to-one) via shared qualification manifest")
PY

python - <<'PY'
from pathlib import Path
import pandas as pd
import difflib

root = Path("outputs/diagnostics/model_qualification/balanced_A1_100")
idx = pd.read_csv(root / "prompt_index.csv")
fail = False
for _, row in idx.iterrows():
    sid = row["sample_id"]
    glm_prompt_path = (root / row["prompt_path"]).resolve()
    can_path = Path("outputs/verification_ablation_1000/A1_overlay_only/prompts") / f"{sid}.txt"
    glm_text = glm_prompt_path.read_text(encoding="utf-8")
    can_text = can_path.read_text(encoding="utf-8")
    if glm_text != can_text:
        fail = True
        print(f"PARITY FAIL {sid}")
        print("".join(difflib.unified_diff(
            can_text.splitlines(True), glm_text.splitlines(True),
            "canonical", "balanced100", n=2,
        ))[:500])
        break
    for token in ["Reliable", "Uncertain", "Unreliable"]:
        assert token in glm_text
    img = (root / row["image_path"]).resolve()
    assert img.is_file(), img
if fail:
    raise SystemExit("PROMPT_PARITY_FAIL")
print("PROMPT_PARITY_PASS: balanced100 prompts byte-identical to canonical A1_overlay_only")
print("Semantic decisions: Reliable / Uncertain / Unreliable (unchanged order)")
print("No confidence/geometry metadata; overlay-only A1 visual")
print("Model-specific wrapping is chat-template only (applied at inference)")
PY

VERIFICATION_ARGS=(
    --model glm_4_6v_flash
    --condition balanced100
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

echo "===== Inference (balanced100 ONLY; no auto-continuation) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "STAGE1_BALANCED100_COMPLETE — hard stop (no A1-1000 / robustness / 5747)"
