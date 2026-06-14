#!/usr/bin/env python3
"""Run mock LVM verification on prepared palm inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvm.mock_verifier import MockVerifier

METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata.csv"
RESULTS_CSV = PROJECT_ROOT / "outputs" / "mock_verification_results.csv"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"

RESULT_COLUMNS = [
    "image_name",
    "palm_id",
    "classification",
    "confidence",
    "bbox_alignment",
    "palm_structure",
    "occlusion_level",
    "reasoning",
    "model_name",
]


def load_model_config(config_path: Path) -> dict:
    """Load model settings from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        print("Run prepare_lvm_inputs.py first.")
        sys.exit(1)

    if not MODEL_CONFIG.exists():
        print(f"Model config not found: {MODEL_CONFIG}")
        sys.exit(1)

    config = load_model_config(MODEL_CONFIG)
    model_name = config.get("mock_model_name", "mock_verifier_v0")

    metadata_df = pd.read_csv(METADATA_CSV)
    verifier = MockVerifier(model_name=model_name)

    print(f"Running mock verification with {model_name}")
    print(f"Found {len(metadata_df)} palm input(s)")

    results: list[dict] = []
    for _, row in metadata_df.iterrows():
        image_path = PROJECT_ROOT / row["lvm_input_path"]
        metadata = row.to_dict()

        result = verifier.verify_image(str(image_path), metadata)
        results.append(result)
        print(
            f"  {row['palm_id']} ({row['image_name']}): "
            f"{result['classification']} (confidence={result['confidence']:.2f})"
        )

    results_df = pd.DataFrame(results)[RESULT_COLUMNS]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    print()
    print("Mock verification complete")
    print(f"  Results saved to: {RESULTS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  Reliable: {(results_df['classification'] == 'reliable').sum()}")
    print(f"  Uncertain: {(results_df['classification'] == 'uncertain').sum()}")
    print(f"  Unreliable: {(results_df['classification'] == 'unreliable').sum()}")


if __name__ == "__main__":
    main()
