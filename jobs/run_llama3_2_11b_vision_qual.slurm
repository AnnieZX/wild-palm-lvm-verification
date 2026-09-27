#!/bin/bash
#SBATCH --job-name=llama32v_s10
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/llama3_2_11b_vision_qual_%j.out
#SBATCH --error=logs/slurm/llama3_2_11b_vision_qual_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Llama 3.2 11B Vision Instruct qualification (engineering only).
#
#   STAGE=sanity10  fixed Stage 0 set outputs/diagnostics/model_qualification/stage0_A1_10/
#
# Default cluster python (/usr/bin/python: transformers 4.57.6, torch 2.8.0+cu128).
#
# Submit from project root:
#   STAGE=sanity10 EXPERIMENT_ID=YYYYMMDD_llama3_2_11b_vision_qual sbatch \
#     --partition=yangGrp --nodelist=lovelace --gres=gpu:L40S:1 \
#     jobs/run_llama3_2_11b_vision_qual.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

STAGE="${STAGE:?Set STAGE=sanity10}"
MODEL="llama3_2_11b_vision"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_llama3_2_11b_vision_qual}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Llama-3.2-11B-Vision-Instruct}"
MODEL_CONFIG="${PROJECT_DIR}/configs/models/${MODEL}.yaml"
PYTHON_BIN="${LLAMA_PYTHON:-/usr/bin/python}"
EXP_ROOT="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}"
STAGE0_INDEX="${PROJECT_DIR}/outputs/diagnostics/model_qualification/stage0_A1_10/prompt_index.csv"

mkdir -p logs/slurm "${EXP_ROOT}"

unset PYTHONPATH || true
export PYTHONPATH="${PROJECT_DIR}"

echo "===== Llama 3.2 11B Vision qualification: ${STAGE} ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Model path:    ${MODEL_PATH}"
echo "Python:        ${PYTHON_BIN}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Start time:    $(date)"
cat "${MODEL_PATH}/DOWNLOAD_META.txt"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" - <<'PY'
import torch, transformers
from transformers import MllamaForConditionalGeneration, MllamaProcessor  # noqa: F401
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
nvidia-smi
echo

# Preflight: audited input contract on the first Stage 0 sample (CPU only).
STAGE0_INDEX="${STAGE0_INDEX}" MODEL_PATH="${MODEL_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from PIL import Image
from transformers import MllamaProcessor
from src.lvm.llama3_2_11b_vision_verifier import build_llama3_2_vision_inputs
from src.verification.jobs import load_verification_jobs

proc = MllamaProcessor.from_pretrained(os.environ["MODEL_PATH"])
job = load_verification_jobs(prompt_index=Path(os.environ["STAGE0_INDEX"]))[0]
prompt = job.prompt_path.read_text(encoding="utf-8")
with Image.open(job.image_path) as im:
    x = build_llama3_2_vision_inputs(proc, im.convert("RGB"), prompt)
ids = x["input_ids"][0].tolist()
text = proc.tokenizer.decode(ids, skip_special_tokens=False)
expected = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    f"<|image|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
assert text == expected, "rendered prompt differs from audited contract"
assert ids.count(128000) == 1 and ids.count(128256) == 1 and text.count(prompt) == 1
for banned in ("Cutting Knowledge", "Today Date", "<|start_header_id|>system"):
    assert banned not in text, banned
print(f"PREFLIGHT_CONTRACT_OK {job.sample_id} tokens={len(ids)} pixel_values={tuple(x['pixel_values'].shape)}")
PY

MEM_LOG="${EXP_ROOT}/gpu_mem_${STAGE}_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader -l 2 > "${MEM_LOG}" &
MEM_PID=$!
trap 'kill ${MEM_PID} 2>/dev/null || true' EXIT

run_one() {
    local condition="$1" prompt_index="$2" results_dir="$3" limit="$4"
    if [[ -f "${results_dir}/results_index.csv" ]]; then
        echo "ERROR: ${results_dir}/results_index.csv exists; choose a new EXPERIMENT_ID." >&2
        exit 1
    fi
    mkdir -p "${results_dir}"
    "${PYTHON_BIN}" -u scripts/run_verification.py \
        --model "${MODEL}" \
        --condition "${condition}" \
        --prompt-index "${prompt_index}" \
        --results-dir "${results_dir}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --batch-size "${BATCH_SIZE}" \
        --model-config "${MODEL_CONFIG}" \
        --model-path "${MODEL_PATH}" \
        --max-new-tokens 512 \
        --limit "${limit}"
}

case "${STAGE}" in
    sanity10)
        run_one stage0 "${STAGE0_INDEX}" "${EXP_ROOT}/stage0" 10
        ;;
    *)
        echo "ERROR: STAGE must be sanity10 (got ${STAGE})" >&2
        exit 1
        ;;
esac

kill "${MEM_PID}" 2>/dev/null || true
echo
echo "Peak GPU memory (MiB): $(awk -F', ' '{gsub(/ MiB/,"",$3); if ($3>m) m=$3} END {print m}' "${MEM_LOG}")"
echo "End time: $(date)"
echo "LLAMA3_2_VISION_${STAGE}_COMPLETE"
