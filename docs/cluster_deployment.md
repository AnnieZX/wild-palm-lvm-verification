# Cluster Deployment Guide

This project separates **local development** (MacBook) from **GPU inference** (Linux cluster).

Local machines are used for preprocessing, prompt design, and mock verification.
Qwen2.5-VL inference should run on a Linux GPU cluster.

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

This installs only the packages needed for GPU inference:

- `torch`, `torchvision`
- `transformers`, `accelerate`, `safetensors`
- `qwen-vl-utils`
- `numpy`, `pandas`, `Pillow`, `pyyaml`, `tqdm`

It does **not** include local-only tools such as `jupyter`, `openai`, or `matplotlib`.

## Step 4: Check the cluster environment

```bash
python scripts/check_cluster_environment.py
```

Expected output includes:

- Python version
- Torch version
- CUDA availability
- CUDA device count
- GPU names (if available)

## Step 5: Run the single-image Qwen test skeleton

First, prepare LVM inputs on your local machine or copy them to the cluster:

```bash
python scripts/prepare_lvm_inputs.py
```

Then on the cluster:

```bash
python scripts/run_qwen_single.py
```

At this stage the script will:

- load the first row from `outputs/lvm_inputs_metadata.csv`
- build the verification prompt
- print image path, palm ID, and prompt length
- attempt to instantiate `QwenVerifier`

Until real inference is enabled, it prints:

```text
Qwen inference has not been enabled yet.
```

## Transferring data to the cluster

Copy these folders/files from local development to the cluster:

```text
data/samples/
outputs/lvm_inputs/
outputs/lvm_inputs_metadata.csv
configs/model.yaml
```

## Enabling real Qwen inference

When ready on the cluster, edit:

```text
src/lvm/qwen_verifier.py
```

Set:

```python
ENABLE_QWEN_INFERENCE = True
```

Then implement model loading and generation inside `_load_model()` and `verify_image()`.

## Future models

The project is designed to support multiple open-source vision-language models:

| Model | Notes |
|-------|-------|
| `Qwen/Qwen2.5-VL-3B-Instruct` | Recommended first cluster model (lighter) |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Higher quality, more GPU memory |
| LLaVA | General-purpose VLM baseline |
| GeoChat | Remote sensing / geospatial specialist |

Update `active_model` in `configs/model.yaml` to switch models once adapters are implemented.

## Recommended workflow

1. **Local MacBook:** preprocessing, overlays, metadata, mock verification
2. **Cluster:** environment check, Qwen single-image test, then batch inference
3. **Local or cluster:** evaluation against human labels (future phase)
