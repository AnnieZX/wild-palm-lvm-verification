#!/usr/bin/env python3
"""
Purpose:
    Compare YOLO prediction counts vs LabelMe palm ground-truth counts per image.

Input:
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON under Raw_Patches

Output:
    - outputs/yolo_analysis/prediction_statistics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PREDICTIONS_FULL_JSON, PREDICTION_STATISTICS_CSV, RAW_PATCHES_ROOT
from src.preprocessing.palm_analyzer import extract_palm_instances
from src.preprocessing.sequential_dataset import find_labelme_json_files
from src.yolo.predictions_io import count_predictions_by_image, load_predictions


def count_ground_truth_palms(json_path: Path) -> int:
    return len(extract_palm_instances(json_path))


def collect_image_names(
    json_files: list[Path],
    prediction_counts: dict[str, int],
) -> list[str]:
    names = {json_path.stem for json_path in json_files}
    names.update(prediction_counts.keys())
    return sorted(names)


def main() -> None:
    if not PREDICTIONS_FULL_JSON.exists():
        print(f"Prediction file not found: {PREDICTIONS_FULL_JSON}")
        sys.exit(1)

    if not RAW_PATCHES_ROOT.exists():
        print(f"Raw_Patches directory not found: {RAW_PATCHES_ROOT}")
        sys.exit(1)

    records = load_predictions(PREDICTIONS_FULL_JSON)
    prediction_counts = count_predictions_by_image(records)
    json_files = find_labelme_json_files(RAW_PATCHES_ROOT)
    json_by_stem = {json_path.stem: json_path for json_path in json_files}

    rows: list[dict[str, int | str]] = []
    for image_name in collect_image_names(json_files, prediction_counts):
        json_path = json_by_stem.get(image_name)
        gt_count = count_ground_truth_palms(json_path) if json_path is not None else 0
        rows.append(
            {
                "image_name": image_name,
                "gt_count": gt_count,
                "prediction_count": prediction_counts.get(image_name, 0),
            }
        )

    df = pd.DataFrame(rows)
    PREDICTION_STATISTICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTION_STATISTICS_CSV, index=False)

    total_images = len(df)
    avg_gt = df["gt_count"].mean() if total_images else 0.0
    avg_pred = df["prediction_count"].mean() if total_images else 0.0

    print("Prediction vs ground-truth statistics")
    print(f"  Predictions: {PREDICTIONS_FULL_JSON}")
    print(f"  Raw_Patches: {RAW_PATCHES_ROOT}")
    print(f"  Output CSV:  {PREDICTION_STATISTICS_CSV}")
    print()
    print(f"total images: {total_images}")
    print(f"average GT palms/image: {avg_gt:.4f}")
    print(f"average predictions/image: {avg_pred:.4f}")
    print()
    print(f"Saved {total_images} rows to {PREDICTION_STATISTICS_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
