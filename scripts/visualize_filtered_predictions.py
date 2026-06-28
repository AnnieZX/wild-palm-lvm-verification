#!/usr/bin/env python3
"""Visualize confidence-filtered and NMS-filtered YOLO predictions on Raw_Patches."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = Path(
    "/deac/csc/yangGrp/cuij/palm/testing/results/yolo_new/"
    "yolo11x_new_datanew-yolo/val/predictions.json"
)
IMAGES_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "filtered_overlay"

CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5

BOX_COLOR = (0, 0, 255)  # BGR red
TEXT_COLOR = (0, 255, 255)  # BGR yellow
BOX_THICKNESS = 2


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
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).stem


def extract_image_id(record: dict[str, Any]) -> str | None:
    for key in ("image_id", "imageId", "image", "file_name", "filename", "imagePath"):
        if key in record:
            image_id = normalize_image_id(record[key])
            if image_id:
                return image_id
    return None


def extract_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
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


def extract_score(record: dict[str, Any]) -> float | None:
    for key in ("score", "confidence", "conf", "probability"):
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


def iou_xywh(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
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

    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def group_predictions_by_image(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
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
                "score": extract_score(record),
            }
        )

    if skipped:
        print(f"Skipped {skipped} record(s) missing image_id or bbox.")
        print()

    return dict(grouped)


def filter_by_confidence(
    predictions: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Keep predictions with score >= threshold."""
    kept: list[dict[str, Any]] = []
    for prediction in predictions:
        score = prediction.get("score")
        if score is None:
            continue
        if score >= threshold:
            kept.append(prediction)
    return kept


def apply_nms(
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Greedy non-maximum suppression on (x, y, w, h) boxes."""
    if not predictions:
        return []

    indexed = list(enumerate(predictions))
    indexed.sort(
        key=lambda item: item[1].get("score") if item[1].get("score") is not None else -1.0,
        reverse=True,
    )

    keep_indices: list[int] = []
    while indexed:
        best_pos, best_item = indexed.pop(0)
        keep_indices.append(best_pos)
        best_bbox = best_item["bbox"]
        indexed = [
            (pos, item)
            for pos, item in indexed
            if iou_xywh(best_bbox, item["bbox"]) < iou_threshold
        ]

    keep_indices.sort()
    return [predictions[index] for index in keep_indices]


def resolve_image_path(image_name: str) -> Path | None:
    direct = IMAGES_ROOT / f"{image_name}.png"
    if direct.exists():
        return direct

    matches = sorted(IMAGES_ROOT.rglob(f"{image_name}.png"))
    if matches:
        return matches[0]

    return None


def draw_predictions(
    image: np.ndarray,
    predictions: list[dict[str, Any]],
) -> np.ndarray:
    overlay = image.copy()

    for prediction in predictions:
        x, y, width, height = prediction["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOR, thickness=BOX_THICKNESS)

        score = prediction.get("score")
        label = f"{score:.3f}" if score is not None else "n/a"
        text_x = max(x1, 0)
        text_y = max(y1 - 6, 16)
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return overlay


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"Prediction file not found: {PREDICTIONS_PATH}")
        sys.exit(1)

    if not IMAGES_ROOT.exists():
        print(f"Images directory not found: {IMAGES_ROOT}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_predictions(PREDICTIONS_PATH)
    grouped = group_predictions_by_image(records)

    print("Filtered YOLO prediction overlays")
    print(f"  Predictions: {PREDICTIONS_PATH}")
    print(f"  Images:      {IMAGES_ROOT}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Confidence threshold: >= {CONFIDENCE_THRESHOLD}")
    print(f"  NMS IoU threshold: {NMS_IOU_THRESHOLD}")
    print()

    saved = 0
    missing_images = 0

    for image_name in sorted(grouped.keys()):
        raw_predictions = grouped[image_name]
        before_filter = len(raw_predictions)

        after_confidence = filter_by_confidence(raw_predictions, CONFIDENCE_THRESHOLD)
        after_conf_count = len(after_confidence)

        after_nms = apply_nms(after_confidence, NMS_IOU_THRESHOLD)
        after_nms_count = len(after_nms)

        print(image_name)
        print(f"  predictions before filtering: {before_filter}")
        print(f"  after confidence filtering:   {after_conf_count}")
        print(f"  after NMS:                    {after_nms_count}")

        image_path = resolve_image_path(image_name)
        if image_path is None:
            print(f"  SKIP: image not found for {image_name}")
            missing_images += 1
            print()
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  SKIP: could not read {image_path}")
            missing_images += 1
            print()
            continue

        overlay = draw_predictions(image, after_nms)
        output_path = OUTPUT_DIR / f"{image_name}_filtered.png"
        cv2.imwrite(str(output_path), overlay)
        print(f"  saved: {output_path.relative_to(PROJECT_ROOT)}")
        saved += 1
        print()

    print("Done")
    print(f"  images with predictions: {len(grouped)}")
    print(f"  overlays saved: {saved}")
    print(f"  missing/unreadable images: {missing_images}")


if __name__ == "__main__":
    main()
