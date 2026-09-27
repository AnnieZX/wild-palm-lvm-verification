#!/usr/bin/env bash
# Submit InternVL3.5-HF A1–A5 @5747 after Stage 1 qualification PASS.
# Usage:
#   EXPERIMENT_ID=20260924_internvl3_5_hf_A1A5_5747 SUBMIT=1 bash jobs/submit_internvl3_5_hf_Ax_5747.sh
# Dry-run (default): prints sbatch commands only.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d)_internvl3_5_hf_A1A5_5747}"
SUBMIT="${SUBMIT:-0}"
PARTITION="${PARTITION:-yangGrp}"
GRES="${GRES:-gpu:L40S:1}"

echo "EXPERIMENT_ID=${EXPERIMENT_ID}"
echo "RESULTS_ROOT=outputs/verification/internvl3_5_hf/${EXPERIMENT_ID}"
echo "PARTITION=${PARTITION} GRES=${GRES}"

if [[ -d "outputs/verification/internvl3_5_hf/${EXPERIMENT_ID}" ]]; then
  echo "ERROR: experiment dir already exists; choose a new EXPERIMENT_ID" >&2
  exit 1
fi

# Preflight: canonical inputs + checkpoint + registry
MODEL_PATH="/deac/csc/yangGrp/luoz23/models/InternVL3_5-8B-HF"
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: missing checkpoint ${MODEL_PATH}" >&2; exit 1; }
[[ -f "configs/models/internvl3_5_hf.yaml" ]] || { echo "ERROR: missing model config" >&2; exit 1; }

python - <<'PY'
import pandas as pd
from pathlib import Path
from src.verification.registry import resolve_registry_key, get_registered_models
from src.config.model_config import normalize_model_key, resolve_model_checkpoint

assert resolve_registry_key("internvl3_5_hf") == "internvl3_5_hf"
assert normalize_model_key("internvl3_5_hf") == "internvl3_5_hf"
assert "internvl3_5_hf" in get_registered_models()
ckpt = resolve_model_checkpoint("internvl3_5_hf")
assert Path(ckpt).exists(), ckpt
print("REGISTRY_OK", ckpt)

conds = {
    "A1": "A1_overlay_only",
    "A2": "A2_overlay_confidence",
    "A3": "A3_overlay_confidence_geometry",
    "A4": "A4_overlay_crop_confidence",
    "A5": "A5_crop_only",
}
a1 = set(pd.read_csv("outputs/verification_ablation_5747/A1_overlay_only/prompt_index.csv").sample_id.astype(str))
assert len(a1) == 5747
for code, folder in conds.items():
    p = Path(f"outputs/verification_ablation_5747/{folder}/prompt_index.csv")
    idx = pd.read_csv(p)
    assert len(idx) == 5747 and idx.sample_id.nunique() == 5747
    assert set(idx.sample_id.astype(str)) == a1
    assert set(idx.condition.unique()) == {folder}
    print(f"INPUT_OK {code} {folder} N=5747")
print("PREFLIGHT_PASS")
PY

mkdir -p logs/slurm

for A in A1 A2 A3 A4 A5; do
  cmd=(sbatch
    --partition="${PARTITION}"
    --gres="${GRES}"
    --job-name="ivl35hf_${A}_5747"
    --output="logs/slurm/internvl3_5_hf_${A}_5747_%j.out"
    --error="logs/slurm/internvl3_5_hf_${A}_5747_%j.err"
    --export=ALL,EXPERIMENT_ID="${EXPERIMENT_ID}",ABLATION="${A}"
    jobs/run_internvl3_5_hf_Ax_5747.slurm
  )
  echo "${cmd[*]}"
  if [[ "${SUBMIT}" == "1" ]]; then
    "${cmd[@]}"
  fi
done

if [[ "${SUBMIT}" != "1" ]]; then
  echo "DRY_RUN only. Re-run with SUBMIT=1 to submit."
fi
