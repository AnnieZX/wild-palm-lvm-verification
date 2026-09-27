#!/usr/bin/env bash
# Prepare (do NOT auto-submit) Phi-4 A1–A5 @5747 after A2 @1000 passes.
# Usage:
#   EXPERIMENT_ID=20260921_phi4_A1A5_5747 bash jobs/submit_phi4_Ax_5747.sh
# Or dry-run (default): prints sbatch commands only.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d)_phi4_A1A5_5747}"
SUBMIT="${SUBMIT:-0}"

echo "EXPERIMENT_ID=${EXPERIMENT_ID}"
echo "RESULTS_ROOT=outputs/verification/phi4_multimodal/${EXPERIMENT_ID}"
if [[ -d "outputs/verification/phi4_multimodal/${EXPERIMENT_ID}" ]]; then
  echo "ERROR: experiment dir already exists; choose a new EXPERIMENT_ID" >&2
  exit 1
fi

for A in A1 A2 A3 A4 A5; do
  cmd=(sbatch
    --partition=yangGrp
    --gres=gpu:L40S:1
    --job-name="phi4_${A}_5747"
    --output="logs/slurm/phi4_${A}_5747_%j.out"
    --error="logs/slurm/phi4_${A}_5747_%j.err"
    --export=ALL,EXPERIMENT_ID="${EXPERIMENT_ID}",ABLATION="${A}"
    jobs/run_phi4_Ax_5747.slurm
  )
  echo "${cmd[*]}"
  if [[ "${SUBMIT}" == "1" ]]; then
    "${cmd[@]}"
  fi
done

if [[ "${SUBMIT}" != "1" ]]; then
  echo "DRY_RUN only. Re-run with SUBMIT=1 to submit."
fi
