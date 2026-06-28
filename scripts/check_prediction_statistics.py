#!/usr/bin/env python3
"""Compare YOLO prediction counts vs LabelMe palm ground-truth counts per image."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.palm_analyzer import extract_palm_instances

PREDICTIONS_PATH = Path(
    "/deac/csc/yangGrp/cuij/palm/testing/results/yolo_new/"
    "yolo11x_new_datanew-yolo/val/predictions.json"
)
RAW_PATCHES_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "prediction_statistics.csv"


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


def count_predictions_by_image(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count valid YOLO predictions grouped by normalized image_id."""
    counts: dict[str, int] = defaultdict(int)
    skipped = 0

    for record in records:
        image_id = extract_image_id(record)
        bbox = extract_bbox(record)
        if image_id is None or bbox is None:
            skipped += 1
            continue
        counts[image_id] += 1

    if skipped:
        print(f"Skipped {skipped} prediction record(s) missing image_id or bbox.")

    return dict(counts)


def find_labelme_json_files(dataset_root: Path) -> list[Path]:
    """Return all LabelMe JSON files under Raw_Patches, sorted by path."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Raw_Patches directory not found: {dataset_root}")
    return sorted(path for path in dataset_root.rglob("*.json") if path.is_file())


def count_ground_truth_palms(json_path: Path) -> int:
    """Count palm instances in one LabelMe JSON file."""
    return len(extract_palm_instances(json_path))


def collect_image_names(
    json_files: list[Path],
    prediction_counts: dict[str, int],
) -> list[str]:
    """Union of JSON stems and prediction image_ids, sorted."""
    names = {json_path.stem for json_path in json_files}
    names.update(prediction_counts.keys())
    return sorted(names)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"Prediction file not found: {PREDICTIONS_PATH}")
        sys.exit(1)

    if not RAW_PATCHES_ROOT.exists():
        print(f"Raw_Patches directory not found: {RAW_PATCHES_ROOT}")
        sys.exit(1)

    records = load_predictions(PREDICTIONS_PATH)
    prediction_counts = count_predictions_by_image(records)

    json_files = find_labelme_json_files(RAW_PATCHES_ROOT)
    json_by_stem = {json_path.stem: json_path for json_path in json_files}

    image_names = collect_image_names(json_files, prediction_counts)
    rows: list[dict[str, int | str]] = []

    for image_name in image_names:
        json_path = json_by_stem.get(image_name)
        gt_count = count_ground_truth_palms(json_path) if json_path is not None else 0
        pred_count = prediction_counts.get(image_name, 0)
        rows.append(
            {
                "image_name": image_name,
                "gt_count": gt_count,
                "prediction_count": pred_count,
            }
        )

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    total_images = len(df)
    avg_gt = df["gt_count"].mean() if total_images else 0.0
    avg_pred = df["prediction_count"].mean() if total_images else 0.0

    print("Prediction vs ground-truth statistics")
    print(f"  Predictions: {PREDICTIONS_PATH}")
    print(f"  Raw_Patches: {RAW_PATCHES_ROOT}")
    print(f"  Output CSV:  {OUTPUT_CSV}")
    print()
    print(f"total images: {total_images}")
    print(f"average GT palms/image: {avg_gt:.4f}")
    print(f"average predictions/image: {avg_pred:.4f}")
    print()
    print(f"Saved {total_images} rows to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
