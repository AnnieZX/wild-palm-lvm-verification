# Qwen2.5-VL Setup

This project uses **Qwen2.5-VL** as the first real open-source vision-language model for palm verification.

## Recommended first model

Start with the smaller instruct model for local debugging:

- `Qwen/Qwen2.5-VL-3B-Instruct`

It is easier to test than the 7B variant and is sufficient for validating the prompt, JSON parsing, and single-image pipeline.

## Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

This installs:

- `transformers`
- `accelerate`
- `torch`
- `torchvision`
- `qwen-vl-utils`

If model loading fails, try upgrading Transformers and Qwen utilities:

```bash
pip install --upgrade transformers accelerate qwen-vl-utils
```

## Run single-image test

Make sure LVM inputs already exist:

```bash
python scripts/prepare_lvm_inputs.py
```

Then run one palm through Qwen2.5-VL:

```bash
python scripts/run_qwen_single.py
```

This script:

- reads the **first row** of `outputs/lvm_inputs_metadata.csv`
- loads the matching image from `outputs/lvm_inputs/`
- sends the verification prompt to Qwen2.5-VL
- saves the raw response to `outputs/raw_responses/qwen_single_raw.txt`
- saves the parsed result to `outputs/qwen_single_result.json`

## What this is for

This is **single-image debugging only**.

It is meant to confirm that:

1. the model loads correctly
2. the prompt works
3. the JSON response can be parsed and validated

Do **not** use this script for full dataset evaluation yet.

## Next step: cluster deployment

After the single-image test works locally or on a GPU machine:

1. verify prompt quality on a few hard cases
2. move to batch inference on the cluster
3. compare Qwen results against mock output and human labels

Cluster deployment should happen **after** this single-image path works reliably.
