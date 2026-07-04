#!/usr/bin/env python3
"""
Purpose:
    Build prompt/input ablation conditions for Qwen2.5-VL verification on 100 YOLO detections.

Input:
    - outputs/verification_dataset/index.csv (first 100 samples)
    - outputs/verification_dataset/images/
    - outputs/verification_dataset/metadata/

Output:
    - outputs/verification_ablation_100/A1_overlay_only/
    - outputs/verification_ablation_100/A2_overlay_confidence/
    - outputs/verification_ablation_100/A3_overlay_confidence_geometry/
    - outputs/verification_ablation_100/A4_overlay_crop_confidence/
    - outputs/verification_ablation_100/ablation_prompt_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    VERIFICATION_ABLATION_DIR,
    VERIFICATION_DATASET_DIR,
)
from src.prompts.ablation_verification_prompt_builder import build_ablation_verification_inputs
from src.prompts.ablation_verification_prompts import ABLATION_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build verification ablation prompts and A4 combined images.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=VERIFICATION_DATASET_DIR,
        help="Verification dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VERIFICATION_ABLATION_DIR,
        help="Ablation output root",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of verification samples to include (default: 100)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(ABLATION_CONDITIONS),
        default=list(ABLATION_CONDITIONS),
        help="Ablation conditions to build",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Verification dataset not found: {args.dataset_dir}")
        print("Run scripts/generate_verification_dataset.py first.")
        sys.exit(1)

    print("Verification ablation prompt builder")
    print(f"  Dataset:      {args.dataset_dir}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Sample count: {args.sample_count}")
    print(f"  Conditions:   {', '.join(args.conditions)}")
    print()

    summary_df = build_ablation_verification_inputs(
        verification_dataset_dir=args.dataset_dir,
        output_root=args.output_dir,
        sample_count=args.sample_count,
        conditions=tuple(args.conditions),
    )

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(
            f"{row['condition']}: {row['sample_count']} samples | "
            f"prompts={row['prompt_dir']} | images={row['image_dir']}"
        )
    print()
    print(f"Saved summary: {args.output_dir / 'ablation_prompt_summary.csv'}")
    print()
    print("Run inference per condition, e.g.:")
    print(
        "  python scripts/run_verification_inference.py "
        f"--dataset-dir {args.output_dir}/A1_overlay_only "
        f"--results-dir outputs/verification_ablation_results/A1_overlay_only"
    )


if __name__ == "__main__":
    main()
