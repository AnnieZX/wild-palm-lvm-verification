#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs

JOB_ID="$(sbatch jobs/qwen_verification_ablation.slurm | awk '{print $NF}')"

echo "Submitted verification ablation job: ${JOB_ID}"
echo "Monitor with:"
echo "  squeue -u \$USER"
