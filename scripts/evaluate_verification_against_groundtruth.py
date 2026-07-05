#!/usr/bin/env python3
"""
Purpose:
    Evaluate verification ablation results by matching YOLO detections to LabelMe GT.

Input:
    - outputs/verification_ablation_results/ (A1–A5 result JSON per sample)
    - outputs/verification_dataset/index.csv
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON under Raw_Patches

Output:
    - outputs/evaluation/A1_evaluation.csv … A5_evaluation.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    EVALUATION_DIR,
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_ABLATION_RESULTS_DIR,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.preprocessing.json_parser import load_json
from src.yolo.predictions_io import extract_score, group_predictions_by_image, iou_xywh, load_predictions

IOU_THRESHOLD = 0.5
CONDITION_CODE_PATTERN = re.compile(r"^(A\d+)")

OUTPUT_COLUMNS = [
    "image_name",
    "sample_id",
    "ablation",
    "yolo_bbox",
    "gt_bbox",
    "max_iou",
    "matched_gt",
    "yolo_confidence",
    "verification_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match YOLO verification samples to LabelMe GT and export evaluation CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=VERIFICATION_ABLATION_RESULTS_DIR,
        help="Verification ablation results root",
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
        help="LabelMe JSON root (Raw_Patches)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATION_DIR,
        help="Directory for per-ablation evaluation CSV files",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
        help="IoU threshold for GT match (default: 0.5)",
    )
    return parser.parse_args()


def discover_ablation_dirs(results_root: Path) -> list[Path]:
    """Return ablation condition directories containing sample result JSON files."""
    if not results_root.exists():
        return []
    return sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and any(path.glob("sample_*.json"))
    )


def condition_code(condition_dir: Path) -> str:
    """Map folder name like A1_overlay_only to A1."""
    match = CONDITION_CODE_PATTERN.match(condition_dir.name)
    if match:
        return match.group(1)
    return condition_dir.name


def bbox_to_json(bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return ""
    x, y, width, height = bbox
    return json.dumps([round(x, 4), round(y, 4), round(width, 4), round(height, 4)])


def resolve_labelme_json(annotations_root: Path, image_name: str) -> Path | None:
    """Find LabelMe JSON for an image stem."""
    candidates = [
        annotations_root / f"{image_name}.json",
        annotations_root / image_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(annotations_root.rglob(f"{image_name}.json"))
    return matches[0] if matches else None


def extract_gt_palm_bboxes(json_path: Path) -> list[tuple[float, float, float, float]]:
    """
    Extract axis-aligned GT palm bboxes from rotation palm shapes.

    Uses label == palm and shape_type == rotation; ignores center/end annotations.
    Returns list of (x, y, width, height).
    """
    data = load_json(json_path)
    grouped: dict[int, list[dict[str, Any]]] = {}

    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        group_id = shape.get("group_id")
        if group_id is None:
            continue
        grouped.setdefault(int(group_id), []).append(shape)

    bboxes: list[tuple[float, float, float, float]] = []
    for group_id in sorted(grouped):
        palm_shapes = [
            shape
            for shape in grouped[group_id]
            if shape.get("label") == "palm" and shape.get("shape_type") == "rotation"
        ]
        if not palm_shapes:
            continue

        points = palm_shapes[0].get("points", [])
        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
        if not xs or not ys:
            continue

        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        bboxes.append((xmin, ymin, xmax - xmin, ymax - ymin))

    return bboxes


def find_yolo_prediction(
    image_name: str,
    target_bbox: tuple[float, float, float, float],
    predictions_by_image: dict[str, list[dict[str, Any]]],
) -> tuple[tuple[float, float, float, float] | None, float | None]:
    """Locate the YOLO prediction that best matches the verification sample bbox."""
    predictions = predictions_by_image.get(image_name, [])
    if not predictions:
        return None, None

    best_iou = -1.0
    best_bbox: tuple[float, float, float, float] | None = None
    best_score: float | None = None

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


def match_yolo_to_gt(
    yolo_bbox: tuple[float, float, float, float],
    gt_bboxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[tuple[float, float, float, float] | None, float, bool]:
    """Return best GT match (xywh), max IoU, and whether IoU >= threshold."""
    if not gt_bboxes:
        return None, 0.0, False

    best_iou = 0.0
    best_gt: tuple[float, float, float, float] | None = None
    for gt_bbox in gt_bboxes:
        overlap = iou_xywh(yolo_bbox, gt_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_gt = gt_bbox

    return best_gt, best_iou, best_iou >= iou_threshold


def load_verification_labels(condition_dir: Path) -> dict[str, str]:
    """Load sample_id -> verification decision from ablation result JSON files."""
    labels: dict[str, str] = {}
    for path in sorted(condition_dir.glob("sample_*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get("sample_id", path.stem))
        decision = str(record.get("decision", "") or "").strip()
        labels[sample_id] = decision
    return labels


def index_bbox_from_row(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["bbox_x"]),
        float(row["bbox_y"]),
        float(row["bbox_width"]),
        float(row["bbox_height"]),
    )


def evaluate_ablation_condition(
    *,
    condition_dir: Path,
    index_df: pd.DataFrame,
    predictions_by_image: dict[str, list[dict[str, Any]]],
    annotations_root: Path,
    gt_cache: dict[str, list[tuple[float, float, float, float]]],
    iou_threshold: float,
) -> pd.DataFrame:
    """Build evaluation rows for one ablation condition."""
    ablation_name = condition_dir.name
    verification_labels = load_verification_labels(condition_dir)

    rows: list[dict[str, Any]] = []

    for _, sample in index_df.iterrows():
        sample_id = str(sample["sample_id"])
        image_name = str(sample["image_name"])
        index_bbox = index_bbox_from_row(sample)

        yolo_bbox, yolo_confidence = find_yolo_prediction(
            image_name,
            index_bbox,
            predictions_by_image,
        )
        if yolo_bbox is None:
            yolo_bbox = index_bbox
            yolo_confidence = float(sample["confidence"]) if pd.notna(sample.get("confidence")) else None

        json_path = resolve_labelme_json(annotations_root, image_name)
        if json_path is None:
            gt_bboxes: list[tuple[float, float, float, float]] = []
        elif image_name not in gt_cache:
            gt_cache[image_name] = extract_gt_palm_bboxes(json_path)
            gt_bboxes = gt_cache[image_name]
        else:
            gt_bboxes = gt_cache[image_name]

        gt_bbox, max_iou, matched_gt = match_yolo_to_gt(yolo_bbox, gt_bboxes, iou_threshold)

        rows.append(
            {
                "image_name": image_name,
                "sample_id": sample_id,
                "ablation": ablation_name,
                "yolo_bbox": bbox_to_json(yolo_bbox),
                "gt_bbox": bbox_to_json(gt_bbox),
                "max_iou": round(max_iou, 4),
                "matched_gt": matched_gt,
                "yolo_confidence": yolo_confidence if yolo_confidence is not None else "",
                "verification_label": verification_labels.get(sample_id, ""),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def print_condition_summary(df: pd.DataFrame, condition_code_name: str) -> None:
    total = len(df)
    matched = int(df["matched_gt"].sum()) if total else 0
    unmatched = total - matched
    avg_iou = float(df["max_iou"].mean()) if total else 0.0
    confidence = pd.to_numeric(df["yolo_confidence"], errors="coerce")
    avg_conf = float(confidence.mean()) if not confidence.dropna().empty else 0.0

    print(f"\n{condition_code_name} ({df['ablation'].iloc[0] if total else 'n/a'})")
    print(f"  samples:             {total}")
    print(f"  matched detections:  {matched}")
    print(f"  unmatched detections:{unmatched}")
    print(f"  average IoU:         {avg_iou:.4f}")
    print(f"  average YOLO conf:   {avg_conf:.4f}")


def main() -> None:
    args = parse_args()

    if not args.index_csv.exists():
        print(f"Verification index not found: {args.index_csv}")
        sys.exit(1)

    if not args.predictions.exists():
        print(f"Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not args.annotations_root.exists():
        print(f"LabelMe annotations root not found: {args.annotations_root}")
        sys.exit(1)

    ablation_dirs = discover_ablation_dirs(args.results_dir)
    if not ablation_dirs:
        print(f"No ablation result folders found under: {args.results_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(args.index_csv)
    predictions_by_image = group_predictions_by_image(load_predictions(args.predictions))
    gt_cache: dict[str, list[tuple[float, float, float, float]]] = {}

    print("Verification vs ground-truth evaluation")
    print(f"  Results:      {args.results_dir}")
    print(f"  Index:        {args.index_csv}")
    print(f"  Predictions:  {args.predictions}")
    print(f"  Annotations:  {args.annotations_root}")
    print(f"  Output:       {args.output_dir}")
    print(f"  IoU threshold:{args.iou_threshold}")

    for condition_dir in ablation_dirs:
        code = condition_code(condition_dir)
        eval_df = evaluate_ablation_condition(
            condition_dir=condition_dir,
            index_df=index_df,
            predictions_by_image=predictions_by_image,
            annotations_root=args.annotations_root,
            gt_cache=gt_cache,
            iou_threshold=args.iou_threshold,
        )

        output_path = args.output_dir / f"{code}_evaluation.csv"
        eval_df.to_csv(output_path, index=False)
        print_condition_summary(eval_df, code)
        print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()
