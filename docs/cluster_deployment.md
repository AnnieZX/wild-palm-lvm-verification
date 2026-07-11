# Cluster Deployment Guide

This project separates **local development** (MacBook) from **GPU inference** (Linux cluster).

Local machines are used for preprocessing, prompt design, and lightweight testing.
VLM verification inference runs on a Linux GPU cluster (SLURM).

Current production model: **Qwen2.5-VL** (`--model qwen2_5_vl`). See [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md).

---

## Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd wild-palm-lvm-verification
```

## Step 2: Create a conda environment

```bash
conda create -n palm-lvm python=3.11
conda activate palm-lvm
```

## Step 3: Install cluster dependencies

```bash
pip install -r requirements_cluster.txt
```

This installs GPU inference packages: `torch`, `transformers`, `qwen-vl-utils`, etc.

Local-only tools (`jupyter`, `matplotlib`) are in `requirements.txt`.

## Step 4: Check the cluster environment

```bash
python scripts/pipeline/check_cluster_environment.py
```

Expected output: Python version, Torch version, CUDA availability, GPU names.

## Step 5: Download model weights (one-time)

```bash
sbatch jobs/qwen_download.slurm
```

Checkpoint path is configured in `configs/models/qwen2_5_vl.yaml` (legacy fallback: `configs/model.yaml`).

## Step 6: Run verification

### Single condition

```bash
python scripts/run_verification.py \
  --model qwen2_5_vl \
  --prompt-index outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv \
  --results-dir outputs/verification/qwen2_5_vl/my_run/A1 \
  --batch-size 4
```

### Full A1–A5 experiment

```bash
SAMPLE_SIZE=1000 bash jobs/submit_qwen_ablation.sh
```

Set `MODEL=qwen2_5_vl` (default) or a future registry key when additional adapters are implemented.

---

## Transferring data to the cluster

Copy or generate on the cluster:

```text
outputs/full_inference/predictions_full.json
outputs/verification_dataset/
outputs/verification_ablation_<N>/
configs/models/qwen2_5_vl.yaml
```

Raw patches on the DEAC cluster: `/deac/csc/yangGrp/cuij/palm/Raw_Patches/`

---

## Output layout

```
outputs/verification/<model_key>/<experiment_id>/A1/
outputs/evaluation/<model_key>/<experiment_id>/A1/
```

Legacy pre-freeze paths under `outputs/verification/qwen/` remain readable.

See [`outputs/README.md`](../outputs/README.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Future models

Adding LLaVA, Gemma 4, or Qwen3-VL requires only verifier + adapter + config + registry entry.

Integration design: [`MULTI_MODEL_INTEGRATION_PLAN.md`](MULTI_MODEL_INTEGRATION_PLAN.md)

---

## Recommended workflow

1. **Local:** dataset prep, ablation prompt generation, evaluation scripts
2. **Cluster:** model download, A1–A5 verification via SLURM
3. **Either:** evaluation, metrics, visualization

Main README: [`../README.md`](../README.md)
