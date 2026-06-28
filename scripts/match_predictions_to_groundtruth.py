#!/usr/bin/env python3
"""Match YOLO predictions to LabelMe palm ground-truth boxes via pairwise IoU."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.palm_analyzer import (
    PalmInstance,
    extract_palm_instances_in_annotation_order,
)

PREDICTIONS_PATH = Path(
    "/deac/csc/yangGrp/cuij/palm/testing/results/yolo_new/"
    "yolo11x_new_datanew-yolo/val/predictions.json"
)
RAW_PATCHES_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "yolo_gt_matches.csv"

IOU_THRESHOLD = 0.5


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load predictions.json and return a flat list of prediction records."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected predictions root type: {type(data)}")

    if isinstance(data.get("predictions"), list):
        return [record for record in data["predictions"] if isinstance(record, dict)]

    if isinstance(data.get("annotations"), list):
        images = {
            str(image["id"]): image
            for image in data.get("images", [])
            if isinstance(image, dict) and "id" in image
        }
        flat: list[dict[str, Any]] = []
        for annotation in data["annotations"]:
            if not isinstance(annotation, dict):
                continue
            record = dict(annotation)
            image_ref = record.get("image_id")
            if image_ref is not None and str(image_ref) in images:
                image_info = images[str(image_ref)]
                for key in ("file_name", "filename", "imagePath"):
                    if key in image_info:
                        record.setdefault("file_name", image_info[key])
            flat.append(record)
        return flat

    if all(isinstance(value, dict) for value in data.values()):
        return list(data.values())

    raise ValueError("Could not interpret predictions.json structure.")


def normalize_image_id(value: Any) -> str | None:
    """Return a stem-style image id such as 100_0003_0001_1."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).stem


def extract_image_id(record: dict[str, Any]) -> str | None:
    """Read image_id from common YOLO / COCO export field names."""
    for key in ("image_id", "imageId", "image", "file_name", "filename", "imagePath"):
        if key in record:
            image_id = normalize_image_id(record[key])
            if image_id:
                return image_id
    return None


def extract_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Read bbox as [x, y, width, height]."""
    bbox = record.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]

    keys = ("x", "y", "width", "height")
    if all(key in record for key in keys):
        return (
            float(record["x"]),
            float(record["y"]),
            float(record["width"]),
            float(record["height"]),
        )

    return None


def extract_confidence(record: dict[str, Any]) -> float | None:
    """Read detector confidence from common field names."""
    for key in ("confidence", "score", "conf", "probability"):
        if key not in record:
            continue
        value = record[key]
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def palm_instance_bbox(palm: PalmInstance) -> tuple[float, float, float, float]:
    """Return GT bbox as (x, y, width, height)."""
    return palm.bbox_x, palm.bbox_y, palm.bbox_width, palm.bbox_height


def bbox_to_list(bbox: tuple[float, float, float, float]) -> list[float]:
    """Serialize bbox with rounded values for CSV output."""
    return [round(value, 2) for value in bbox]


def iou_xywh(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU for axis-aligned boxes in (x, y, width, height) format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def group_predictions_by_image(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group valid YOLO predictions by normalized image_id."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for record in records:
        image_id = extract_image_id(record)
        bbox = extract_bbox(record)
        if image_id is None or bbox is None:
            skipped += 1
            continue
        grouped[image_id].append(
            {
                "bbox": bbox,
                "confidence": extract_confidence(record),
            }
        )

    if skipped:
        print(f"Skipped {skipped} prediction record(s) missing image_id or bbox.")

    return dict(grouped)


def find_labelme_json_files(dataset_root: Path) -> list[Path]:
    """Return all LabelMe JSON files under Raw_Patches, sorted by path."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Raw_Patches directory not found: {dataset_root}")
    return sorted(path for path in dataset_root.rglob("*.json") if path.is_file())


def match_gt_to_best_prediction(
    gt_bbox: tuple[float, float, float, float],
    predictions: list[dict[str, Any]],
) -> tuple[tuple[float, float, float, float] | None, float | None, float]:
    """
    Assign one GT palm to the prediction with highest IoU.

    Returns (pred_bbox, confidence, best_iou).
    """
    if not predictions:
        return None, None, 0.0

    best_iou = 0.0
    best_pred_bbox: tuple[float, float, float, float] | None = None
    best_confidence: float | None = None

    for prediction in predictions:
        pred_bbox = prediction["bbox"]
        overlap = iou_xywh(gt_bbox, pred_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_pred_bbox = pred_bbox
            best_confidence = prediction.get("confidence")

    return best_pred_bbox, best_confidence, best_iou


def match_image(
    image_name: str,
    json_path: Path,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build match rows for every GT palm in one image."""
    palms = extract_palm_instances_in_annotation_order(json_path)
    rows: list[dict[str, Any]] = []

    for index, palm in enumerate(palms, start=1):
        gt_bbox = palm_instance_bbox(palm)
        pred_bbox, confidence, best_iou = match_gt_to_best_prediction(gt_bbox, predictions)
        matched = best_iou >= IOU_THRESHOLD

        rows.append(
            {
                "image_name": image_name,
                "palm_id": f"palm_{index:02d}",
                "gt_bbox": json.dumps(bbox_to_list(gt_bbox)),
                "pred_bbox": json.dumps(bbox_to_list(pred_bbox)) if pred_bbox else "",
                "confidence": confidence if confidence is not None else "",
                "IoU": round(best_iou, 4),
                "matched": matched,
            }
        )

    return rows


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"Prediction file not found: {PREDICTIONS_PATH}")
        sys.exit(1)

    if not RAW_PATCHES_ROOT.exists():
        print(f"Raw_Patches directory not found: {RAW_PATCHES_ROOT}")
        sys.exit(1)

    records = load_predictions(PREDICTIONS_PATH)
    predictions_by_image = group_predictions_by_image(records)
    json_files = find_labelme_json_files(RAW_PATCHES_ROOT)

    rows: list[dict[str, Any]] = []
    for json_path in json_files:
        image_name = json_path.stem
        predictions = predictions_by_image.get(image_name, [])
        rows.extend(match_image(image_name, json_path, predictions))

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    total_gt = len(df)
    matched_count = int(df["matched"].sum()) if total_gt else 0
    match_rate = (matched_count / total_gt * 100) if total_gt else 0.0

    print("YOLO-to-ground-truth matching")
    print(f"  Predictions: {PREDICTIONS_PATH}")
    print(f"  Raw_Patches: {RAW_PATCHES_ROOT}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print(f"  Output CSV:  {OUTPUT_CSV}")
    print()
    print(f"images processed: {len(json_files)}")
    print(f"GT palms: {total_gt}")
    print(f"matched (IoU >= {IOU_THRESHOLD}): {matched_count} ({match_rate:.2f}%)")
    print()
    print(f"Saved {total_gt} rows to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
