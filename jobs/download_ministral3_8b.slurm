#!/bin/bash
#SBATCH --job-name=ministral3_dl
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/ministral3_8b_download_%j.out
#SBATCH --error=logs/slurm/ministral3_8b_download_%j.err
#
# Official Mistral BF16 checkpoint, pinned revision (Apache-2.0; NOT gated).
# consolidated.safetensors / params.json / tekken.json are the mistral-native
# duplicate of the same weights (vLLM format) and are not needed by Transformers.

set -euo pipefail

MODEL_ID="mistralai/Ministral-3-8B-Instruct-2512-BF16"
REVISION="${REVISION:-f6fae9795746f63c9be8344932f01275f3c63734}"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/Ministral-3-8B-Instruct-2512-BF16"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs/slurm "${MODEL_DIR}"

module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source /deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4/bin/activate
export PYTHONNOUSERSITE=1

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start: $(date)"
echo "Model ID: ${MODEL_ID} @ ${REVISION}"
echo "Target: ${MODEL_DIR}"

MODEL_ID="${MODEL_ID}" REVISION="${REVISION}" MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

model_id = os.environ["MODEL_ID"]
revision = os.environ["REVISION"]
model_dir = os.environ["MODEL_DIR"]

info = HfApi().model_info(model_id, revision=revision)
print(f"HF repo: {model_id}")
print(f"Revision/commit: {info.sha}")
print(f"gated: {info.gated}")
assert info.sha == revision, (info.sha, revision)

path = snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=model_dir,
    ignore_patterns=["consolidated.safetensors", "params.json", "tekken.json"],
)
print(f"Download complete: {path}")
Path(model_dir, "DOWNLOAD_META.txt").write_text(
    f"repo_id={model_id}\nrevision={info.sha}\ngated={info.gated}\n"
    f"local_dir={model_dir}\n"
    "ignored=consolidated.safetensors,params.json,tekken.json\n",
    encoding="utf-8",
)
PY

du -sh "${MODEL_DIR}"
echo "End: $(date)"
