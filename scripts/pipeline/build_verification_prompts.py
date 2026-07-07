#!/usr/bin/env python3
"""
Purpose:
    Build Qwen2.5-VL text prompts for every verification dataset sample.

Input:
    - Verification dataset with images/, metadata/, and index.csv
      (default: outputs/verification_dataset/)

Output:
    - outputs/verification_dataset/prompts/sample_XXXXXX.txt (one per sample)
    - outputs/verification_dataset/prompt_index.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import VERIFICATION_DATASET_DIR
from src.prompts.verification_prompt_builder import build_prompts_for_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qwen verification prompts from a verification dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=VERIFICATION_DATASET_DIR,
        help="Verification dataset root (contains images/, metadata/, index.csv)",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Prompt output directory (default: <dataset-dir>/prompts/)",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="Dataset index CSV (default: <dataset-dir>/index.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    index_csv = args.index_csv or dataset_dir / "index.csv"
    prompts_dir = args.prompts_dir or dataset_dir / "prompts"

    if not index_csv.exists():
        print(f"Index CSV not found: {index_csv}")
        print("Run scripts/pipeline/generate_verification_dataset.py first.")
        sys.exit(1)

    print("Verification prompt builder")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Index:   {index_csv}")
    print(f"  Output:  {prompts_dir}")
    print()

    prompt_index = build_prompts_for_dataset(
        dataset_dir=dataset_dir,
        prompts_dir=prompts_dir,
        index_csv=index_csv,
    )

    prompt_index_path = dataset_dir / "prompt_index.csv"
    prompt_index.to_csv(prompt_index_path, index=False)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Prompts written: {len(prompt_index)}")
    print(f"Saved prompts:   {prompts_dir}/")
    print(f"Saved index:     {prompt_index_path}")


if __name__ == "__main__":
    main()
