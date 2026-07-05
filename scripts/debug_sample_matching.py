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
from src.preprocessing.gt_palm_bboxes import extract_gt_palm_bboxes
from src.preprocessing.verification_dataset import resolve_patch_image
from src.utils.labelme_paths import resolve_labelme_json
from src.utils.verification_index import find_yolo_prediction, index_bbox_from_row
from src.visualization.verification_matching_debug import render_sample_matching_debug
from src.yolo.gt_matching import greedy_match_bboxes_to_gt
from src.yolo.predictions_io import group_predictions_by_image, iou_xywh, load_predictions

IOU_THRESHOLD = 0.5


def resolve_yolo_bbox_for_sample(
    row: pd.Series,
    predictions_by_image: dict[str, list],
) -> tuple[tuple[float, float, float, float], float | None]:
    image_name = str(row["image_name"])
    index_bbox = index_bbox_from_row(row)
    yolo_bbox, yolo_confidence = find_yolo_prediction(image_name, index_bbox, predictions_by_image)
    if yolo_bbox is None:
        yolo_bbox = index_bbox
        yolo_confidence = float(row["confidence"]) if pd.notna(row.get("confidence")) else None
    return yolo_bbox, yolo_confidence


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
    yolo_bbox, yolo_confidence = resolve_yolo_bbox_for_sample(row, predictions_by_image)

    json_path = resolve_labelme_json(args.annotations_root, image_name)
    gt_bboxes = extract_gt_palm_bboxes(json_path) if json_path is not None else []

    image_samples = index_df[index_df["image_name"].astype(str) == image_name]
    sample_ids: list[str] = []
    yolo_bboxes: list[tuple[float, float, float, float]] = []
    for _, sample_row in image_samples.iterrows():
        sample_bbox, _ = resolve_yolo_bbox_for_sample(sample_row, predictions_by_image)
        sample_ids.append(str(sample_row["sample_id"]))
        yolo_bboxes.append(sample_bbox)

    greedy_matches = greedy_match_bboxes_to_gt(yolo_bboxes, gt_bboxes, args.iou_threshold)
    sample_index = sample_ids.index(args.sample_id)
    match = greedy_matches[sample_index]
    matched_index = match.gt_index if match.matched_gt else match.gt_index
    max_iou = match.max_iou
    matched_gt = match.matched_gt
    per_gt_iou = [iou_xywh(yolo_bbox, gt_bbox) for gt_bbox in gt_bboxes]

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
            marker = "  <-- greedy match" if match.matched_gt and index == matched_index else ""
            print(f"   [{index}] {format_bbox(gt_bbox)}  IoU={iou_text}{marker}")
    print(f"5. max IoU:               {max_iou:.4f}")
    print(f"6. matched GT index:      {matched_index if matched_gt else 'none (greedy one-to-one)'}")
    print(f"7. matched_gt:            {matched_gt}")
    print(f"   IoU threshold:          {args.iou_threshold}")
    print(f"   image detections:       {len(sample_ids)} (greedy matching across image)")

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
