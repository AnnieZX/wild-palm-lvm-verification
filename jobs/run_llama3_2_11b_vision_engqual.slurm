#!/bin/bash
#SBATCH --job-name=llama32v_engqual
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/llama3_2_11b_vision_engqual_%j.out
#SBATCH --error=logs/slurm/llama3_2_11b_vision_engqual_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Llama 3.2 11B Vision Instruct ENGINEERING qualification (non-L40S hardware).
# Not part of the scientific L40S comparison.
#
#   1. sanity-10: fixed Stage 0 set outputs/diagnostics/model_qualification/stage0_A1_10/
#   2. automatic sanity gate
#   3. only if the gate passes: A1..A5 smoke, first SMOKE_LIMIT (4) samples of
#      outputs/verification_ablation_1000/<condition>/prompt_index.csv
#
# Default cluster python (/usr/bin/python: transformers 4.57.6, torch 2.8.0+cu128).
#
# Submit from project root:
#   EXPERIMENT_ID=YYYYMMDD_llama3_2_11b_vision_engineering_qualification sbatch \
#     --partition=gpu_small --gres=gpu:1 --constraint="h200_141|a100_80|a100_40" \
#     jobs/run_llama3_2_11b_vision_engqual.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

MODEL="llama3_2_11b_vision"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d)_llama3_2_11b_vision_engineering_qualification}"
BATCH_SIZE=1
SMOKE_LIMIT=4
MODEL_PATH="/deac/csc/yangGrp/luoz23/models/Llama-3.2-11B-Vision-Instruct"
MODEL_CONFIG="${PROJECT_DIR}/configs/models/${MODEL}.yaml"
PYTHON_BIN="/usr/bin/python"
EXP_ROOT="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}"
STAGE0_INDEX="${PROJECT_DIR}/outputs/diagnostics/model_qualification/stage0_A1_10/prompt_index.csv"

mkdir -p logs/slurm "${EXP_ROOT}"

unset PYTHONPATH || true
export PYTHONPATH="${PROJECT_DIR}"

stamp() { while IFS= read -r line; do printf '[%(%F %T)T] %s\n' -1 "${line}"; done; }

echo "===== Llama 3.2 11B Vision ENGINEERING qualification ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "Partition:     ${SLURM_JOB_PARTITION:-unknown}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Model path:    ${MODEL_PATH}"
echo "Python:        ${PYTHON_BIN}"
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "${EXP_ROOT}/gpu_type_${SLURM_JOB_ID:-local}.txt"
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

MEM_LOG="${EXP_ROOT}/gpu_mem_${SLURM_JOB_ID:-local}.csv"
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
        --limit "${limit}" 2>&1 | stamp
}

echo "===== STAGE: sanity10 ====="
run_one stage0 "${STAGE0_INDEX}" "${EXP_ROOT}/stage0" 10

echo "===== SANITY GATE ====="
if ! RESULTS_DIR="${EXP_ROOT}/stage0" "${PYTHON_BIN}" - <<'PY'
import csv, json, os, sys
from collections import Counter
from pathlib import Path

d = Path(os.environ["RESULTS_DIR"])
rows = list(csv.DictReader(open(d / "results_index.csv")))
recs = [json.loads((d / r["result_path"]).read_text()) for r in rows]
raws = [r["raw_response"] for r in recs]
fails = []
if len(recs) != 10:
    fails.append(f"expected 10 records, got {len(recs)}")
n_inf = sum(bool(r["inference_error"]) for r in recs)
n_parse_ok = sum(bool(r["decision"]) and not r["parse_error"] for r in recs)
if n_inf:
    fails.append(f"{n_inf} inference errors")
if any(not s.strip() for s in raws):
    fails.append("empty generation(s)")
if n_parse_ok < 8:
    fails.append(f"parse success {n_parse_ok}/10 < 8")
echo_markers = ("You are an expert reviewer", "Wild palm characteristics", "Return ONLY valid JSON")
if any(m in s for s in raws for m in echo_markers):
    fails.append("prompt echo")
leak_markers = ("Cutting Knowledge", "Today Date", "<|start_header_id|>", "<|eot_id|>", "<|image|>")
if any(m in s for s in raws for m in leak_markers):
    fails.append("system/date/special-token leakage")
if any(len(s) > 6000 for s in raws):
    fails.append("overlong generation (possible truncation / runaway)")
if len(raws) == 10 and len(set(raws)) == 1:
    fails.append("10 identical raw outputs")
if any(s.lower() in ("nan", "inf") or "\ufffd" in s for s in raws):
    fails.append("corrupted output")
print("decisions:", dict(Counter(r["decision"] or "<none>" for r in recs)))
print(f"parse_ok={n_parse_ok}/10 inference_errors={n_inf}")
if fails:
    print("SANITY_GATE_FAIL:", "; ".join(fails))
    sys.exit(3)
print("SANITY_GATE_PASS")
PY
then
    echo "Sanity gate failed; smoke NOT run."
    kill "${MEM_PID}" 2>/dev/null || true
    echo "Peak GPU memory (MiB): $(awk -F', ' '{gsub(/ MiB/,"",$3); if ($3>m) m=$3} END {print m}' "${MEM_LOG}")"
    echo "LLAMA3_2_VISION_ENGQUAL_SANITY_FAIL"
    exit 3
fi

echo "===== STAGE: A1-A5 smoke (${SMOKE_LIMIT} per condition) ====="
for cond in A1_overlay_only A2_overlay_confidence A3_overlay_confidence_geometry \
            A4_overlay_crop_confidence A5_crop_only; do
    code="${cond%%_*}"
    run_one "${code}" \
        "${PROJECT_DIR}/outputs/verification_ablation_1000/${cond}/prompt_index.csv" \
        "${EXP_ROOT}/smoke/${code}" "${SMOKE_LIMIT}"
done

kill "${MEM_PID}" 2>/dev/null || true
echo
echo "Peak GPU memory (MiB): $(awk -F', ' '{gsub(/ MiB/,"",$3); if ($3>m) m=$3} END {print m}' "${MEM_LOG}")"
echo "End time: $(date)"
echo "LLAMA3_2_VISION_ENGQUAL_COMPLETE"
