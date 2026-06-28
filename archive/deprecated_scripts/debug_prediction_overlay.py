#!/usr/bin/env python3
"""Visualize YOLO predictions on Raw_Patches images (standalone debug tool)."""

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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug_prediction_overlay"

MAX_IMAGES = 10
BBOX_COLOR = (0, 255, 0)  # BGR green
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
    """
    Read bbox as [x, y, width, height].

    Also accepts dict keys x/y/width/height when present.
    """
    bbox = record.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, width, height = (float(value) for value in bbox)
        return x, y, width, height

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


def group_predictions_by_image(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group flat prediction records by normalized image_id."""
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
        print(f"Skipped {skipped} record(s) missing image_id or bbox.")

    return dict(grouped)


def resolve_image_path(images_root: Path, image_id: str) -> Path:
    """Return Raw_Patches/{image_id}.png."""
    return images_root / f"{image_id}.png"


def draw_predictions(
    image: np.ndarray,
    predictions: list[dict[str, Any]],
) -> np.ndarray:
    """Draw all predicted bboxes and confidence labels onto a copy of the image."""
    overlay = image.copy()

    for prediction in predictions:
        x, y, width, height = prediction["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), BBOX_COLOR, thickness=BOX_THICKNESS)

        confidence = prediction.get("confidence")
        if confidence is None:
            label = "?"
        else:
            label = f"{confidence:.2f}"

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


def confidence_list(predictions: list[dict[str, Any]]) -> list[float | None]:
    """Return confidences in prediction order."""
    return [prediction.get("confidence") for prediction in predictions]


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

    if not grouped:
        print("No valid predictions with image_id and bbox were found.")
        sys.exit(1)

    selected_ids = sorted(grouped.keys())[:MAX_IMAGES]

    print("YOLO prediction overlay debug")
    print(f"  Predictions: {PREDICTIONS_PATH}")
    print(f"  Images:      {IMAGES_ROOT}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Rendering first {len(selected_ids)} image(s)")
    print()

    for image_id in selected_ids:
        predictions = grouped[image_id]
        image_path = resolve_image_path(IMAGES_ROOT, image_id)

        if not image_path.exists():
            print(f"image_id: {image_id}")
            print(f"  MISSING image: {image_path}")
            print(f"  boxes: {len(predictions)}")
            print(f"  confidences: {confidence_list(predictions)}")
            print()
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"image_id: {image_id}")
            print(f"  ERROR: could not read image: {image_path}")
            print(f"  boxes: {len(predictions)}")
            print(f"  confidences: {confidence_list(predictions)}")
            print()
            continue

        overlay = draw_predictions(image, predictions)
        output_path = OUTPUT_DIR / f"{image_id}_overlay.png"
        cv2.imwrite(str(output_path), overlay)

        confidences = confidence_list(predictions)
        print(f"image_id: {image_id}")
        print(f"  boxes: {len(predictions)}")
        print(f"  confidences: {confidences}")
        print(f"  saved: {output_path.relative_to(PROJECT_ROOT)}")
        print()


if __name__ == "__main__":
    main()
