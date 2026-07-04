#!/bin/bash
#SBATCH --job-name=qwen_verif_ablation
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/qwen_verif_ablation_%j.out
#SBATCH --error=logs/qwen_verif_ablation_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Verification input ablation: A1–A5 on 100 YOLO detections.
# Cluster settings reused from archive/old_labelme_ablation/jobs/qwen_ablation_100.slurm
#
# Prerequisites:
#   python scripts/build_ablation_verification_prompts.py
#
# Submit from the project root:
#   bash scripts/submit_qwen_ablation.sh
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
ABLATION_ROOT="${PROJECT_DIR}/outputs/verification_ablation_100"
RESULTS_ROOT="${PROJECT_DIR}/outputs/verification_ablation_results"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Project directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p "${RESULTS_ROOT}/A1_overlay_only"
mkdir -p "${RESULTS_ROOT}/A2_overlay_confidence"
mkdir -p "${RESULTS_ROOT}/A3_overlay_confidence_geometry"
mkdir -p "${RESULTS_ROOT}/A4_overlay_crop_confidence"
mkdir -p "${RESULTS_ROOT}/A5_crop_only"

echo "Python: $(which python)"
python --version

echo
echo "=== nvidia-smi ==="
nvidia-smi

for condition in \
    A1_overlay_only \
    A2_overlay_confidence \
    A3_overlay_confidence_geometry \
    A4_overlay_crop_confidence \
    A5_crop_only
do
    PROMPT_INDEX="${ABLATION_ROOT}/${condition}/prompt_index.csv"
    if [[ ! -f "${PROMPT_INDEX}" ]]; then
        echo "ERROR: Prompt index not found: ${PROMPT_INDEX}"
        echo "Run scripts/build_ablation_verification_prompts.py first."
        exit 1
    fi
done

echo
echo "=== A1: overlay only ==="
python scripts/run_verification_inference.py \
  --prompt-index outputs/verification_ablation_100/A1_overlay_only/prompt_index.csv \
  --results-dir outputs/verification_ablation_results/A1_overlay_only \
  --limit 100 \
  --batch-size 2

echo
echo "=== A2: overlay + confidence ==="
python scripts/run_verification_inference.py \
  --prompt-index outputs/verification_ablation_100/A2_overlay_confidence/prompt_index.csv \
  --results-dir outputs/verification_ablation_results/A2_overlay_confidence \
  --limit 100 \
  --batch-size 2

echo
echo "=== A3: overlay + confidence + geometry ==="
python scripts/run_verification_inference.py \
  --prompt-index outputs/verification_ablation_100/A3_overlay_confidence_geometry/prompt_index.csv \
  --results-dir outputs/verification_ablation_results/A3_overlay_confidence_geometry \
  --limit 100 \
  --batch-size 2

echo
echo "=== A4: overlay + crop + confidence ==="
python scripts/run_verification_inference.py \
  --prompt-index outputs/verification_ablation_100/A4_overlay_crop_confidence/prompt_index.csv \
  --results-dir outputs/verification_ablation_results/A4_overlay_crop_confidence \
  --limit 100 \
  --batch-size 2

echo
echo "=== A5: crop only + confidence ==="
python scripts/run_verification_inference.py \
  --prompt-index outputs/verification_ablation_100/A5_crop_only/prompt_index.csv \
  --results-dir outputs/verification_ablation_results/A5_crop_only \
  --limit 100 \
  --batch-size 2

echo
echo "End time: $(date)"
echo "Verification ablation inference complete."
echo "Results root: ${RESULTS_ROOT}/"
echo "Stdout log: ${PROJECT_DIR}/logs/qwen_verif_ablation_${SLURM_JOB_ID}.out"
echo "Stderr log: ${PROJECT_DIR}/logs/qwen_verif_ablation_${SLURM_JOB_ID}.err"
