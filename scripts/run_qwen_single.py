#!/usr/bin/env python3
"""Run Qwen2.5-VL verification on a single palm input for debugging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvm.qwen_verifier import QwenVerifier

METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata.csv"
RESULT_JSON = PROJECT_ROOT / "outputs" / "qwen_single_result.json"
RAW_RESPONSE_PATH = PROJECT_ROOT / "outputs" / "raw_responses" / "qwen_single_raw.txt"


def main() -> None:
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        print("Run prepare_lvm_inputs.py first.")
        sys.exit(1)

    metadata_df = pd.read_csv(METADATA_CSV)
    if metadata_df.empty:
        print("Metadata CSV is empty.")
        sys.exit(1)

    row = metadata_df.iloc[0]
    image_path = PROJECT_ROOT / row["lvm_input_path"]
    metadata = row.to_dict()

    print("Single-image Qwen2.5-VL test")
    print(f"  Image: {image_path.relative_to(PROJECT_ROOT)}")
    print(f"  Palm ID: {metadata.get('palm_id')}")
    print()

    try:
        verifier = QwenVerifier()
    except RuntimeError as error:
        print(error)
        sys.exit(1)

    try:
        result = verifier.verify_image(str(image_path), metadata)
    except Exception as error:
        print("Qwen verification failed.")
        print(error)
        sys.exit(1)

    raw_response = result.get("raw_response", "")
    RAW_RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_RESPONSE_PATH.write_text(raw_response, encoding="utf-8")

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_JSON.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print()
    print("=== Raw model response ===")
    print(raw_response)

    print()
    print("=== Parsed / validated response ===")
    if "parse_error" in result:
        print(f"Parse error: {result['parse_error']}")
        print("Raw response was saved, but JSON validation failed.")
    else:
        validated_view = {
            key: result[key]
            for key in [
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
            if key in result
        }
        print(json.dumps(validated_view, indent=2))

    print()
    print("Saved files:")
    print(f"  {RESULT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  {RAW_RESPONSE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
