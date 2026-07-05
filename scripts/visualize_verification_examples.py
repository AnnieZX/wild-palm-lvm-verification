#!/usr/bin/env python3
"""
Purpose:
    Visualize random verification evaluation samples for publication.

Input:
    - outputs/evaluation/A*_evaluation.csv
    - Raw patch PNGs and LabelMe JSON under Raw_Patches

Output:
    - outputs/visualization/{sample_id}_{ablation}.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EVALUATION_DIR, RAW_PATCHES_ROOT, VISUALIZATION_DIR
from src.preprocessing.verification_dataset import resolve_patch_image
from src.preprocessing.verification_visualization import (
    find_gt_center_for_bbox,
    parse_bbox_json,
    render_verification_example,
)

EVALUATION_CSV_PATTERN = re.compile(r"^(A\d+)_evaluation\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize random verification evaluation examples.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=EVALUATION_DIR,
        help="Directory containing A*_evaluation.csv files",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="Raw patch PNG root",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="LabelMe JSON root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATION_DIR,
        help="Directory for visualization PNG files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of random samples to visualize (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Optional ablation code (e.g. A1) to restrict evaluation CSV",
    )
    return parser.parse_args()


def discover_evaluation_csvs(evaluation_dir: Path) -> list[Path]:
    if not evaluation_dir.exists():
        return []
    return sorted(
        path
        for path in evaluation_dir.glob("*_evaluation.csv")
        if EVALUATION_CSV_PATTERN.match(path.name)
    )


def resolve_labelme_json(annotations_root: Path, image_name: str) -> Path | None:
    candidates = [
        annotations_root / f"{image_name}.json",
        annotations_root / image_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(annotations_root.rglob(f"{image_name}.json"))
    return matches[0] if matches else None


def load_evaluation_pool(evaluation_dir: Path, ablation: str | None) -> pd.DataFrame:
    csv_paths = discover_evaluation_csvs(evaluation_dir)
    if ablation:
        csv_paths = [path for path in csv_paths if path.name.startswith(f"{ablation}_evaluation.csv")]
    if not csv_paths:
        raise FileNotFoundError(f"No evaluation CSV files found under {evaluation_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    pool = pd.concat(frames, ignore_index=True)
    if pool.empty:
        raise ValueError("Evaluation pool is empty.")
    return pool


def main() -> None:
    args = parse_args()

    try:
        pool = load_evaluation_pool(args.evaluation_dir, args.ablation)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        sys.exit(1)

    if not args.images_root.exists():
        print(f"Images root not found: {args.images_root}")
        sys.exit(1)

    sample_count = min(args.count, len(pool))
    selected = pool.sample(n=sample_count, random_state=args.seed).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Verification example visualization")
    print(f"  Evaluation:  {args.evaluation_dir}")
    print(f"  Images:      {args.images_root}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Samples:     {sample_count} / {len(pool)} (seed={args.seed})")
    print()

    saved = 0
    for _, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        image_name = str(row["image_name"])
        ablation = str(row.get("ablation", "unknown"))

        image_path = resolve_patch_image(args.images_root, image_name)
        if image_path is None:
            print(f"  SKIP {sample_id}: image not found for {image_name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  SKIP {sample_id}: could not read {image_path}")
            continue

        yolo_bbox = parse_bbox_json(row.get("yolo_bbox"))
        gt_bbox = parse_bbox_json(row.get("gt_bbox"))

        json_path = resolve_labelme_json(args.annotations_root, image_name)
        gt_center = None
        if json_path is not None:
            reference_bbox = gt_bbox if gt_bbox is not None else yolo_bbox
            gt_center = find_gt_center_for_bbox(json_path, reference_bbox)

        confidence_value = row.get("yolo_confidence")
        try:
            yolo_confidence = float(confidence_value) if pd.notna(confidence_value) and str(confidence_value).strip() else None
        except (TypeError, ValueError):
            yolo_confidence = None

        try:
            iou = float(row.get("max_iou", 0.0))
        except (TypeError, ValueError):
            iou = 0.0

        rendered = render_verification_example(
            image,
            yolo_bbox=yolo_bbox,
            gt_bbox=gt_bbox,
            gt_center=gt_center,
            yolo_confidence=yolo_confidence,
            iou=iou,
            verification_label=str(row.get("verification_label", "") or ""),
            sample_id=sample_id,
            ablation=ablation,
        )

        output_path = args.output_dir / f"{sample_id}_{ablation}.png"
        cv2.imwrite(str(output_path), rendered, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        saved += 1
        print(f"  saved {output_path.name}")

    print()
    print(f"Saved {saved} visualization(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
