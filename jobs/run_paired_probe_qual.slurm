#!/bin/bash
#SBATCH --job-name=paired_probe_qual
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/paired_probe_%j.out
#SBATCH --error=logs/slurm/paired_probe_%j.err
#
# Technical qualification on the fixed GT-independent paired probe.
#
# Probe: sample_000001..sample_000100 under outputs/verification_ablation_100/<A1..A5>/
# (first 100 detections by index order; selection does not use GT). The same 100
# sample IDs are run in all five conditions; inputs are byte-identical to the
# corresponding @5747 inputs. A1's first DETERMINISM_N samples are re-run in a
# second pass to check deterministic generation.
#
# Submit from project root:
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,MODEL=qwen3_vl_4b,EXPERIMENT_ID=20261001_qwen3_vl_4b_paired_probe100 \
#     --job-name=qwen3_vl_4b_probe \
#     --output=logs/slurm/qwen3_vl_4b_probe_%j.out \
#     --error=logs/slurm/qwen3_vl_4b_probe_%j.err \
#     jobs/run_paired_probe_qual.slurm
#
# The verdict is written by scripts/evaluate_paired_probe_qualification.py.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/jobs/lib/model_runtime.sh"

MODEL="$(canonicalize_model_key "${MODEL:?Set MODEL}")"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d)_${MODEL}_paired_probe100}"
BATCH_SIZE="${BATCH_SIZE:-$(model_default_batch_size "${MODEL}")}"
MODEL_PATH="${MODEL_PATH:-$(model_default_checkpoint "${MODEL}")}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_DIR}/configs/models/${MODEL}.yaml}"
DETERMINISM_N="${DETERMINISM_N:-10}"
PROBE_ROOT="${PROJECT_DIR}/outputs/verification_ablation_100"
EXP_ROOT="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}"
REPORT_DIR="${PROJECT_DIR}/outputs/diagnostics/model_qualification/paired_probe/${MODEL}/${EXPERIMENT_ID}"

if [[ -e "${EXP_ROOT}" ]]; then
    echo "ERROR: ${EXP_ROOT} exists; choose a new EXPERIMENT_ID." >&2
    exit 1
fi
mkdir -p logs/slurm "${EXP_ROOT}/_run_metadata" "${REPORT_DIR}"

echo "===== Paired-probe technical qualification ====="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Partition:   ${SLURM_JOB_PARTITION:-}"
echo "Model:       ${MODEL} (family $(model_family "${MODEL}"))"
echo "Checkpoint:  ${MODEL_PATH}"
echo "Config:      ${MODEL_CONFIG}"
echo "Experiment:  ${EXPERIMENT_ID}"
echo "Start:       $(date)"

activate_model_environment "${MODEL}" "${PROJECT_DIR}"
which python
python -c 'import transformers, torch; print("transformers", transformers.__version__); print("torch", torch.__version__)'
nvidia-smi
cat "${MODEL_PATH}/DOWNLOAD_META.txt"

GIT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
export WILD_PALM_GIT_COMMIT="${GIT_COMMIT}"
echo "Git commit:  ${GIT_COMMIT}"
cat > "${EXP_ROOT}/_run_metadata/probe_${SLURM_JOB_ID:-local}.json" <<JSON
{
  "model_key": "${MODEL}",
  "experiment_id": "${EXPERIMENT_ID}",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "slurm_partition": "${SLURM_JOB_PARTITION:-}",
  "node": "${SLURMD_NODENAME:-}",
  "git_commit": "${GIT_COMMIT}",
  "git_dirty_tracked_files": $(git -C "${PROJECT_DIR}" status --porcelain -- src scripts jobs configs | wc -l),
  "checkpoint": "${MODEL_PATH}",
  "model_config": "${MODEL_CONFIG}",
  "probe_root": "${PROBE_ROOT}",
  "determinism_n": ${DETERMINISM_N}
}
JSON

GPU_MEM_LOG="${EXP_ROOT}/_run_metadata/gpu_mem_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader -l 2 > "${GPU_MEM_LOG}" &
GPU_MEM_PID=$!
trap 'kill ${GPU_MEM_PID} 2>/dev/null || true' EXIT

run_probe() {
    local code="$1" folder="$2" results_dir="$3" limit="$4"
    python -u scripts/run_verification.py \
        --model "${MODEL}" \
        --condition "${code}" \
        --prompt-index "${PROBE_ROOT}/${folder}/prompt_index.csv" \
        --results-dir "${results_dir}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --batch-size "${BATCH_SIZE}" \
        --model-config "${MODEL_CONFIG}" \
        --model-path "${MODEL_PATH}" \
        --limit "${limit}"
}

for code in A1 A2 A3 A4 A5; do
    folder="$(condition_dir_name "${code}")"
    echo "===== Probe ${code} (${folder}) $(date) ====="
    run_probe "${code}" "${folder}" "${EXP_ROOT}/${code}" 100
done
echo "===== Determinism re-run: A1 first ${DETERMINISM_N} $(date) ====="
run_probe A1 A1_overlay_only "${EXP_ROOT}/determinism_A1" "${DETERMINISM_N}"

kill "${GPU_MEM_PID}" 2>/dev/null || true

python -u scripts/evaluate_paired_probe_qualification.py \
    --experiment-dir "${EXP_ROOT}" \
    --probe-root "${PROBE_ROOT}" \
    --gpu-mem-log "${GPU_MEM_LOG}" \
    --model-config "${MODEL_CONFIG}" \
    --output-dir "${REPORT_DIR}"

echo "End: $(date)"
echo "PAIRED_PROBE_COMPLETE ${MODEL} ${EXPERIMENT_ID}"
