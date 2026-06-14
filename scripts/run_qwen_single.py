#!/usr/bin/env python3
"""Prepare and run a single-image Qwen2.5-VL verification test."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvm.prompt_template import build_verification_prompt
from src.lvm.qwen_verifier import QwenVerifier

METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata.csv"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"


def load_model_config(config_path: Path) -> dict:
    """Load model settings from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        print("Run prepare_lvm_inputs.py first.")
        sys.exit(1)

    config = load_model_config(MODEL_CONFIG)
    active_model = config.get("active_model", "Qwen/Qwen2.5-VL-3B-Instruct")

    metadata_df = pd.read_csv(METADATA_CSV)
    if metadata_df.empty:
        print("Metadata CSV is empty.")
        sys.exit(1)

    row = metadata_df.iloc[0]
    image_path = PROJECT_ROOT / row["lvm_input_path"]
    metadata = row.to_dict()
    prompt = build_verification_prompt(metadata)

    print("Single-image Qwen2.5-VL test (cluster-ready skeleton)")
    print(f"  Active model: {active_model}")
    print(f"  Image path: {image_path.relative_to(PROJECT_ROOT)}")
    print(f"  Palm ID: {metadata.get('palm_id')}")
    print(f"  Prompt length: {len(prompt)} characters")
    print()

    try:
        verifier = QwenVerifier(model_name=active_model)
        result = verifier.verify_image(str(image_path), metadata)
        print("Qwen inference completed.")
        print(result)
    except NotImplementedError:
        print("Qwen inference has not been enabled yet.")
        print("Enable real inference in src/lvm/qwen_verifier.py on the cluster.")


if __name__ == "__main__":
    main()
