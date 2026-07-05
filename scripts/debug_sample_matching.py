#!/usr/bin/env python3
"""
Purpose:
    Debug GT/YOLO matching for a single verification sample on the cluster.

Input:
    - outputs/verification_dataset/index.csv
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON + PNG under Raw_Patches

Example:
    python scripts/debug_sample_matching.py --sample-id sample_000102
    python scripts/debug_sample_matching.py --sample-id sample_000102 --save-image
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
    VISUALIZATION_DIR,
)
from src.preprocessing.verification_dataset import resolve_patch_image
from src.preprocessing.verification_matching_debug import render_sample_matching_debug
from src.preprocessing.verification_visualization import extract_gt_palm_bboxes
from src.yolo.predictions_io import extract_score, group_predictions_by_image, iou_xywh, load_predictions

IOU_THRESHOLD = 0.5


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


def index_bbox_from_row(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["bbox_x"]),
        float(row["bbox_y"]),
        float(row["bbox_width"]),
        float(row["bbox_height"]),
    )


def find_yolo_prediction(
    image_name: str,
    target_bbox: tuple[float, float, float, float],
    predictions_by_image: dict[str, list],
) -> tuple[tuple[float, float, float, float] | None, float | None]:
    predictions = predictions_by_image.get(image_name, [])
    if not predictions:
        return None, None

    best_iou = -1.0
    best_bbox = None
    best_score = None
    for prediction in predictions:
        bbox = prediction.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        pred_bbox = tuple(float(v) for v in bbox)
        overlap = iou_xywh(target_bbox, pred_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_bbox = pred_bbox
            best_score = extract_score(prediction)
    return best_bbox, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug GT/YOLO matching for one verification sample.",
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help="Verification sample id (e.g. sample_000102)",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=VERIFICATION_DATASET_INDEX_CSV,
        help="Verification dataset index CSV",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_FULL_JSON,
        help="YOLO predictions JSON",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="LabelMe JSON root",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="Raw patch PNG root",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
        help="IoU threshold for matched_gt (default: 0.5)",
    )
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="Save debug image with GT/YOLO boxes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATION_DIR,
        help="Directory for debug image output",
    )
    return parser.parse_args()


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    x, y, width, height = bbox
    return f"[x={x:.2f}, y={y:.2f}, w={width:.2f}, h={height:.2f}]"


def match_yolo_to_gt_with_index(
    yolo_bbox: tuple[float, float, float, float],
    gt_bboxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[int | None, float, bool, list[float]]:
    """Return matched GT index, max IoU, matched flag, and per-GT IoU list."""
    if not gt_bboxes:
        return None, 0.0, False, []

    per_gt_iou: list[float] = []
    best_iou = 0.0
    best_index: int | None = None

    for index, gt_bbox in enumerate(gt_bboxes):
        overlap = iou_xywh(yolo_bbox, gt_bbox)
        per_gt_iou.append(overlap)
        if overlap > best_iou:
            best_iou = overlap
            best_index = index

    matched = best_iou >= iou_threshold
    return best_index, best_iou, matched, per_gt_iou


def main() -> None:
    args = parse_args()

    if not args.index_csv.exists():
        print(f"Index CSV not found: {args.index_csv}")
        sys.exit(1)

    index_df = pd.read_csv(args.index_csv)
    matches = index_df[index_df["sample_id"].astype(str) == args.sample_id]
    if matches.empty:
        print(f"Sample not found in index: {args.sample_id}")
        sys.exit(1)

    row = matches.iloc[0]
    image_name = str(row["image_name"])
    index_bbox = index_bbox_from_row(row)

    predictions_by_image = group_predictions_by_image(load_predictions(args.predictions))
    yolo_bbox, yolo_confidence = find_yolo_prediction(image_name, index_bbox, predictions_by_image)
    if yolo_bbox is None:
        yolo_bbox = index_bbox
        yolo_confidence = float(row["confidence"]) if pd.notna(row.get("confidence")) else None

    json_path = resolve_labelme_json(args.annotations_root, image_name)
    gt_bboxes = extract_gt_palm_bboxes(json_path) if json_path is not None else []

    matched_index, max_iou, matched_gt, per_gt_iou = match_yolo_to_gt_with_index(
        yolo_bbox,
        gt_bboxes,
        args.iou_threshold,
    )

    print("Sample matching debug")
    print("=" * 60)
    print(f"1. image_name:           {image_name}")
    print(f"2. annotation json path:  {json_path if json_path else 'NOT FOUND'}")
    print(f"3. yolo bbox:             {format_bbox(yolo_bbox)}")
    if yolo_confidence is not None:
        print(f"   yolo confidence:       {yolo_confidence:.4f}")
    print(f"   index bbox:            {format_bbox(index_bbox)}")
    print(f"4. all GT bboxes ({len(gt_bboxes)}):")
    if not gt_bboxes:
        print("   (none)")
    else:
        for index, gt_bbox in enumerate(gt_bboxes):
            iou_text = f"{per_gt_iou[index]:.4f}" if index < len(per_gt_iou) else "n/a"
            marker = "  <-- best match" if index == matched_index else ""
            print(f"   [{index}] {format_bbox(gt_bbox)}  IoU={iou_text}{marker}")
    print(f"5. max IoU:               {max_iou:.4f}")
    print(f"6. matched GT index:      {matched_index if matched_index is not None else 'none'}")
    print(f"7. matched_gt:            {matched_gt}")
    print(f"   IoU threshold:          {args.iou_threshold}")

    if args.save_image:
        image_path = resolve_patch_image(args.images_root, image_name)
        if image_path is None:
            print(f"\nERROR: PNG not found for {image_name} under {args.images_root}")
            sys.exit(1)

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"\nERROR: Could not read image: {image_path}")
            sys.exit(1)

        rendered = render_sample_matching_debug(
            image,
            yolo_bbox=yolo_bbox,
            gt_bboxes=gt_bboxes,
            matched_gt_index=matched_index,
            max_iou=max_iou,
            sample_id=args.sample_id,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"debug_{args.sample_id}.png"
        cv2.imwrite(str(output_path), rendered, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"\nSaved debug image: {output_path}")


if __name__ == "__main__":
    main()
