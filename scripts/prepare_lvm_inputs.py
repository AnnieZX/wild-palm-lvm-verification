#!/usr/bin/env python3
"""Prepare per-palm overlay images and metadata for LVM verification."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.lvm_input_builder import build_lvm_inputs

IMAGE_DIR = PROJECT_ROOT / "data" / "samples" / "images"
JSON_DIR = PROJECT_ROOT / "data" / "samples" / "json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lvm_inputs"
METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata.csv"


def main() -> None:
    if not IMAGE_DIR.exists():
        print(f"Image folder not found: {IMAGE_DIR}")
        sys.exit(1)

    if not JSON_DIR.exists():
        print(f"JSON folder not found: {JSON_DIR}")
        sys.exit(1)

    print(f"Building LVM inputs from {IMAGE_DIR}")
    build_lvm_inputs(
        image_dir=IMAGE_DIR,
        json_dir=JSON_DIR,
        output_dir=OUTPUT_DIR,
        metadata_csv=METADATA_CSV,
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
