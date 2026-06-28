#!/usr/bin/env python3
"""Test prompt building and LVM response schema validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvm.prompt_template import build_verification_prompt
from src.lvm.response_schema import parse_json_response, validate_lvm_response

METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata.csv"


def main() -> None:
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        print("Run prepare_lvm_inputs.py first.")
        sys.exit(1)

    metadata_df = pd.read_csv(METADATA_CSV)
    first_row = metadata_df.iloc[0].to_dict()

    print("=== Prompt for first palm ===")
    prompt = build_verification_prompt(first_row)
    print(prompt)

    print("=== Sample fake response ===")
    fake_response = {
        "palm_id": first_row["palm_id"],
        "classification": "Uncertain",
        "confidence": "0.72",
        "bbox_alignment": "partial",
        "palm_structure": "partial",
        "occlusion_level": "medium",
        "reasoning": "The crown is visible but endpoint support is limited.",
    }
    print(json.dumps(fake_response, indent=2))

    print()
    print("=== Validated response ===")
    validated = validate_lvm_response(fake_response)
    print(json.dumps(validated, indent=2))

    print()
    print("=== Parse JSON from fenced raw text ===")
    raw_text = """Here is my answer:
```json
{
  "palm_id": "palm_01",
  "classification": "reliable",
  "confidence": 0.91,
  "bbox_alignment": "good",
  "palm_structure": "clear",
  "occlusion_level": "low",
  "reasoning": "Clear palm crown aligned with the bounding box."
}
```
"""
    parsed = parse_json_response(raw_text)
    cleaned = validate_lvm_response(parsed)
    print(json.dumps(cleaned, indent=2))


if __name__ == "__main__":
    main()
