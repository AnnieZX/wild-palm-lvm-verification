#!/usr/bin/env python3
"""Summarize YOLO prediction counts, confidence, and per-image distribution."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PRED_FILE = Path(
    "/deac/csc/yangGrp/cuij/palm/testing/results/yolo_new/"
    "yolo11x_new_datanew-yolo/val/predictions.json"
)


def main() -> None:
    if not PRED_FILE.exists():
        print(f"Prediction file not found: {PRED_FILE}")
        sys.exit(1)

    with PRED_FILE.open(encoding="utf-8") as file:
        preds = json.load(file)

    if not isinstance(preds, list):
        print(f"Expected a JSON list, got {type(preds).__name__}")
        sys.exit(1)

    per_image: dict[str, list[dict]] = defaultdict(list)
    for prediction in preds:
        if not isinstance(prediction, dict):
            continue
        image_id = prediction.get("image_id")
        if image_id is None:
            continue
        per_image[str(image_id)].append(prediction)

    print("=" * 60)
    print("Prediction Statistics")
    print("=" * 60)

    print(f"Total prediction boxes: {len(preds)}")
    print(f"Images with predictions: {len(per_image)}")

    if not per_image:
        print("No per-image predictions found.")
        return

    counts = [len(values) for values in per_image.values()]

    print(f"Average boxes/image: {mean(counts):.2f}")
    print(f"Max boxes/image: {max(counts)}")
    print(f"Min boxes/image: {min(counts)}")

    print()

    scores = [prediction["score"] for prediction in preds if "score" in prediction]

    print("Confidence")
    if scores:
        print(f"Mean : {mean(scores):.4f}")
        print(f"Max  : {max(scores):.4f}")
        print(f"Min  : {min(scores):.4f}")
    else:
        print("No score field found in predictions.")

    print()

    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
    for threshold in thresholds:
        count = sum(score >= threshold for score in scores)
        print(f"score >= {threshold:.1f}: {count}")

    print()
    print("=" * 60)
    print("Top 20 images by prediction count")
    print("=" * 60)

    ranking = sorted(
        ((image_id, len(values)) for image_id, values in per_image.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    for image_id, count in ranking[:20]:
        print(f"{image_id:<25} {count}")

    print()
    print("=" * 60)
    print("Prediction count distribution")
    print("=" * 60)

    bins = {
        "1-5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51-100": 0,
        ">100": 0,
    }

    for count in counts:
        if count <= 5:
            bins["1-5"] += 1
        elif count <= 10:
            bins["6-10"] += 1
        elif count <= 20:
            bins["11-20"] += 1
        elif count <= 50:
            bins["21-50"] += 1
        elif count <= 100:
            bins["51-100"] += 1
        else:
            bins[">100"] += 1

    for label, value in bins.items():
        print(f"{label:>8}: {value}")

    print()
    print("=" * 60)
    print("First 10 prediction examples")
    print("=" * 60)

    for prediction in preds[:10]:
        print(prediction)


if __name__ == "__main__":
    main()
