#!/bin/bash
#SBATCH --job-name=vlm_verify
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/verify_%j.out
#SBATCH --error=logs/slurm/verify_%j.err
#
# Generic local VLM verification job (ONE GPU via Slurm allocation).
#
# Do NOT hardcode CUDA_VISIBLE_DEVICES. Use the GPU Slurm assigns.
#
# Preferred submit (partition/gres recovered from successful historical jobs):
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,MODEL=qwen2_5_vl,CONDITION=A1,LIMIT=20,EXPERIMENT_ID=... \
#     --job-name=qwen_A1_smoke20 \
#     --output=logs/slurm/qwen_A1_smoke20_%j.out \
#     --error=logs/slurm/qwen_A1_smoke20_%j.err \
#     jobs/run_verification.slurm
#
# Or via:
#   DRY_RUN=1 ./scripts/submit_model_ablation.sh qwen2_5_vl --conditions A1 --limit 20
#
# Positional args (optional; env vars preferred):
#   $1 MODEL   $2 CONDITION   $3 LIMIT   $4 EXPERIMENT_ID
#
# Historical model-specific Slurm scripts under jobs/ remain the source of
# truth for completed experiments. This template is additive.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/jobs/lib/model_runtime.sh"

MODEL_RAW="${1:-${MODEL:?Set MODEL (e.g. qwen2_5_vl)}}"
CONDITION="${2:-${CONDITION:?Set CONDITION to A1|A2|A3|A4|A5}}"
LIMIT="${3:-${LIMIT:-1000}}"
EXPERIMENT_ID="${4:-${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_${MODEL_RAW}_${CONDITION}}}"

MODEL="$(canonicalize_model_key "${MODEL_RAW}")"
CONDITION_DIR="$(condition_dir_name "${CONDITION}")"
BATCH_SIZE="${BATCH_SIZE:-$(model_default_batch_size "${MODEL}")}"
RESUME="${RESUME:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
ABLATION_SAMPLE_SIZE="${ABLATION_SAMPLE_SIZE:-1000}"

PROMPT_INDEX="${PROMPT_INDEX:-${PROJECT_DIR}/outputs/verification_ablation_${ABLATION_SAMPLE_SIZE}/${CONDITION_DIR}/prompt_index.csv}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/${CONDITION}}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_DIR}/outputs/evaluation_protocol_v2/${MODEL}/${EXPERIMENT_ID}/${CONDITION}}"
DATASET_INDEX="${DATASET_INDEX:-${PROJECT_DIR}/outputs/verification_dataset/index.csv}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_DIR}/configs/models/${MODEL}.yaml}"

# Prefer explicit local checkpoints (historical Slurm always passed --model-path
# for Phi-4 because YAML model_id is the HF repo id and would otherwise win).
MODEL_PATH="${MODEL_PATH:-$(model_default_checkpoint "${MODEL}")}"

mkdir -p logs/slurm "${RESULTS_DIR}"
if [[ "${RUN_EVAL}" == "1" ]]; then
    mkdir -p "${EVAL_DIR}"
fi

refuse_protected_results_path "${RESULTS_DIR}"
refuse_protected_results_path "${EVAL_DIR}"

if [[ "${RESUME}" != "1" ]] && [[ -f "${RESULTS_DIR}/results_index.csv" ]]; then
    echo "ERROR: ${RESULTS_DIR}/results_index.csv already exists." >&2
    echo "Choose a new EXPERIMENT_ID or set RESUME=1." >&2
    exit 1
fi

echo "===== Generic local verification ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset (Slurm-assigned)}"
echo "Model:         ${MODEL} (from ${MODEL_RAW})"
echo "Condition:     ${CONDITION} (${CONDITION_DIR})"
echo "Limit:         ${LIMIT}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Evaluation:    ${EVAL_DIR}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Resume:        ${RESUME}"
echo "Run eval:      ${RUN_EVAL}"
echo "Start time:    $(date)"
echo

activate_model_environment "${MODEL}" "${PROJECT_DIR}"

which python
python --version
python -c 'import transformers, torch; print("transformers", transformers.__version__); print("torch", torch.__version__)' || true
nvidia-smi
echo

# Phi-4: fail fast on GPUs with active uncorrectable ECC (historical A2 pathology).
if [[ "${MODEL}" == "phi4_multimodal" ]]; then
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
fi

# GLM: assert isolated transformers pin used by completed experiments.
if [[ "${MODEL}" == "glm_4_6v_flash" ]]; then
    python - <<'PY'
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from transformers import Glm4vForConditionalGeneration  # noqa: F401
print("GLM_TRANSFORMERS_PIN_OK", transformers.__version__)
PY
fi

# Molmo2: assert official card pin (transformers==4.57.1) + remote-code classes.
if [[ "${MODEL}" == "molmo2_8b" ]]; then
    python - <<'PY'
import transformers
assert transformers.__version__ == "4.57.1", transformers.__version__
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
import molmo_utils  # noqa: F401
print("MOLMO2_TRANSFORMERS_PIN_OK", transformers.__version__)
PY
fi

# Gemma 4: require transformers >= 5.10.1 (AutoModelForMultimodalLM / gemma4_unified).
if [[ "${MODEL}" == "gemma4" ]]; then
    python - <<'PY'
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: F401

def _ver(s: str) -> tuple[int, ...]:
    parts = []
    for tok in s.split(".")[:3]:
        digits = "".join(ch for ch in tok if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)

assert _ver(transformers.__version__) >= (5, 10, 1), transformers.__version__
print("GEMMA4_TRANSFORMERS_PIN_OK", transformers.__version__)
PY
fi

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ -n "${MODEL_PATH}" && ! -d "${MODEL_PATH}" && ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "ERROR: missing model config: ${MODEL_CONFIG}" >&2
    exit 1
fi

# Run provenance: written next to (not inside) the condition directory.
GIT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY_FILES="$(git -C "${PROJECT_DIR}" status --porcelain -- src scripts jobs configs 2>/dev/null | wc -l)"
GIT_DIRTY_LIST="$(git -C "${PROJECT_DIR}" status --porcelain -- src scripts jobs configs 2>/dev/null | awk '{print $NF}' | paste -sd, -)"
export WILD_PALM_GIT_COMMIT="${GIT_COMMIT}"
RUN_META_DIR="$(dirname "${RESULTS_DIR}")/_run_metadata"
mkdir -p "${RUN_META_DIR}"
RUN_META_JSON="${RUN_META_DIR}/${CONDITION}_${SLURM_JOB_ID:-local}.json"
GPU_MEM_LOG="${RUN_META_DIR}/${CONDITION}_${SLURM_JOB_ID:-local}_gpu_mem.csv"
if [[ "${GIT_DIRTY_FILES}" != "0" ]]; then
    git -C "${PROJECT_DIR}" diff HEAD -- src scripts jobs configs \
        > "${RUN_META_DIR}/${CONDITION}_${SLURM_JOB_ID:-local}_uncommitted.patch" 2>/dev/null || true
fi
python - "${RUN_META_JSON}" <<PY
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import torch, transformers, yaml
cfg = yaml.safe_load(Path("${MODEL_CONFIG}").read_text()) or {}
revision = ""
meta = Path("${MODEL_PATH}") / "DOWNLOAD_META.txt"
if meta.is_file():
    for line in meta.read_text().splitlines():
        if line.startswith("revision="):
            revision = line.split("=", 1)[1].strip()
record = {
    "model_key": "${MODEL}",
    "model_label": cfg.get("model_label", ""),
    "family": cfg.get("family", "${MODEL}"),
    "parameter_size": cfg.get("parameter_size", ""),
    "condition": "${CONDITION}",
    "limit": int("${LIMIT}"),
    "experiment_id": "${EXPERIMENT_ID}",
    "results_dir": "${RESULTS_DIR}",
    "slurm_job_id": "${SLURM_JOB_ID:-local}",
    "slurm_partition": "${SLURM_JOB_PARTITION:-}",
    "node": "${SLURMD_NODENAME:-}",
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    "git_commit": "${GIT_COMMIT}",
    "git_dirty_tracked_files": int("${GIT_DIRTY_FILES}"),
    "git_dirty_paths": [p for p in "${GIT_DIRTY_LIST}".split(",") if p],
    "checkpoint": "${MODEL_PATH}",
    "checkpoint_revision": revision,
    "config_revision": cfg.get("revision", ""),
    "hf_repo": cfg.get("hf_repo", ""),
    "dtype": cfg.get("dtype", ""),
    "attn_implementation_config": cfg.get("attn_implementation"),
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "resume": "${RESUME}" == "1",
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
if record["config_revision"] and record["config_revision"] != revision:
    sys.exit(f"ERROR: checkpoint revision {revision!r} != config revision {record['config_revision']!r}")
Path(sys.argv[1]).write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
PY
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader -l 5 > "${GPU_MEM_LOG}" &
GPU_MEM_PID=$!
trap 'kill ${GPU_MEM_PID} 2>/dev/null || true' EXIT

VERIFICATION_ARGS=(
    --model "${MODEL}"
    --condition "${CONDITION}"
    --limit "${LIMIT}"
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-config "${MODEL_CONFIG}"
)

if [[ -n "${MODEL_PATH}" ]]; then
    VERIFICATION_ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

echo "===== Inference (load model ONCE, then iterate samples) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"
kill "${GPU_MEM_PID}" 2>/dev/null || true
echo "Peak GPU memory (MiB): $(awk -F', ' '{gsub(/ MiB/,"",$3); if ($3+0>m) m=$3+0} END {print m}' "${GPU_MEM_LOG}")"

if [[ "${RUN_EVAL}" == "1" ]]; then
    echo
    echo "===== Canonical evaluation ====="
    python -u scripts/evaluate_verification_against_groundtruth.py \
        --results-dir "${RESULTS_DIR}" \
        --index-csv "${DATASET_INDEX}" \
        --output-dir "${EVAL_DIR}" \
        --condition-code "${CONDITION}"

    python -u scripts/compute_verification_metrics.py \
        --evaluation-dir "${EVAL_DIR}"
fi

echo
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "${MODEL}_${CONDITION}_LIMIT${LIMIT}_COMPLETE"
