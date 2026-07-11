#!/usr/bin/env python3
"""
Purpose:
    Convert YOLO detections into per-detection verification samples (model-agnostic).

Input:
    - Raw patch PNGs (Raw_Patches or --images-root)
    - YOLO predictions JSON (default: outputs/full_inference/predictions_full.json)
    - Confidence threshold (default: 0.5)

Output:
    - outputs/verification_dataset/images/sample_XXXXXX.png
    - outputs/verification_dataset/metadata/sample_XXXXXX.json
    - outputs/verification_dataset/index.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_DIR,
)
from src.preprocessing.verification_dataset import build_verification_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one verification sample per YOLO detection.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_FULL_JSON,
        help="YOLO predictions JSON path",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="Directory containing raw patch PNG files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VERIFICATION_DATASET_DIR,
        help="Verification dataset output directory",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N images (0 to disable)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        print(f"Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not args.images_root.exists():
        print(f"Images directory not found: {args.images_root}")
        sys.exit(1)

    print("Verification dataset generation")
    print(f"  Images:      {args.images_root}")
    print(f"  Predictions: {args.predictions}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Confidence:  >= {args.confidence}")
    if args.workers is not None:
        print(f"  Workers:     {args.workers}")
    print()

    index_df = build_verification_dataset(
        images_root=args.images_root,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        progress_interval=args.progress_interval,
        num_workers=args.workers,
    )

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Samples written: {len(index_df)}")
    if not index_df.empty:
        print(f"Unique source images: {index_df['image_name'].nunique()}")
        print(f"Mean confidence: {index_df['confidence'].mean():.4f}")
    print()
    print(f"Saved images:   {args.output_dir / 'images'}/")
    print(f"Saved metadata: {args.output_dir / 'metadata'}/")
    print(f"Saved index:    {args.output_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
