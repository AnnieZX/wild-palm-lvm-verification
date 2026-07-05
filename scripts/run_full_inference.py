#!/usr/bin/env python3
"""
Purpose:
    Run YOLO11x inference on all Raw_Patches PNGs and save predictions + overlays.

Input:
    - YOLO weights: best.pt (cluster path)
    - Raw_Patches/*.png

Output:
    - outputs/full_inference/predictions_full.json
    - outputs/full_inference/overlays/{image_id}_overlay.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    FULL_INFERENCE_DIR,
    FULL_INFERENCE_OVERLAYS_DIR,
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    YOLO_MODEL_PATH,
)
from src.yolo.predictions_io import find_png_images

CONF_THRESHOLD = 0.001
MAX_DET = 300
PROGRESS_INTERVAL = 50

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
TITLE_COLOR = (0, 255, 255)
BOX_THICKNESS = 2


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def draw_overlay(image, image_name: str, detections: list[dict]) -> object:
    overlay = image.copy()
    cv2.putText(
        overlay, image_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TITLE_COLOR, 2, cv2.LINE_AA,
    )
    for detection in detections:
        x, y, width, height = detection["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + width)), int(round(y + height))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOR, thickness=BOX_THICKNESS)
        score = detection.get("score")
        label = f"{score:.3f}" if score is not None else "n/a"
        cv2.putText(
            overlay, label, (max(x1, 0), max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return overlay


def run_inference_on_image(model, image_path: Path) -> list[dict]:
    image_id = image_path.stem
    results = model.predict(
        source=str(image_path), conf=CONF_THRESHOLD, max_det=MAX_DET, save=False, verbose=False,
    )
    detections: list[dict] = []
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return detections

    result = results[0]
    for box, score, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        x1, y1, x2, y2 = box
        detections.append(
            {
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": xyxy_to_xywh(x1, y1, x2, y2),
                "score": float(score),
            }
        )
    return detections


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install with: pip install ultralytics")
        sys.exit(1)

    if not YOLO_MODEL_PATH.exists():
        print(f"Model weights not found: {YOLO_MODEL_PATH}")
        sys.exit(1)

    image_paths = find_png_images(RAW_PATCHES_ROOT)
    if not image_paths:
        print(f"No PNG images found under: {RAW_PATCHES_ROOT}")
        sys.exit(1)

    FULL_INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    FULL_INFERENCE_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    print("Full YOLO inference on Raw_Patches")
    print(f"  Model:   {YOLO_MODEL_PATH}")
    print(f"  Images:  {RAW_PATCHES_ROOT}")
    print(f"  Output:  {FULL_INFERENCE_DIR}")
    print(f"  conf:    {CONF_THRESHOLD}")
    print(f"  max_det: {MAX_DET}")
    print(f"  PNG count: {len(image_paths)}")
    print()

    model = YOLO(str(YOLO_MODEL_PATH))
    all_predictions: list[dict] = []
    all_scores: list[float] = []

    for index, image_path in enumerate(image_paths, start=1):
        detections = run_inference_on_image(model, image_path)
        all_predictions.extend(detections)
        all_scores.extend(d["score"] for d in detections)

        image = cv2.imread(str(image_path))
        if image is not None:
            overlay = draw_overlay(image, image_path.stem, detections)
            cv2.imwrite(str(FULL_INFERENCE_OVERLAYS_DIR / f"{image_path.stem}_overlay.png"), overlay)

        if index % PROGRESS_INTERVAL == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} images")

    with PREDICTIONS_FULL_JSON.open("w", encoding="utf-8") as file:
        json.dump(all_predictions, file, indent=2)

    total_images = len(image_paths)
    total_detections = len(all_predictions)
    avg_detections = total_detections / total_images if total_images else 0.0
    avg_confidence = mean(all_scores) if all_scores else 0.0

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections/image: {avg_detections:.4f}")
    print(f"Average confidence: {avg_confidence:.6f}")
    print()
    print(f"score >= 0.5:  {sum(s >= 0.5 for s in all_scores)}")
    print(f"score >= 0.25: {sum(s >= 0.25 for s in all_scores)}")
    print(f"score >= 0.1:  {sum(s >= 0.1 for s in all_scores)}")
    print()
    print(f"Saved predictions: {PREDICTIONS_FULL_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Saved overlays:    {FULL_INFERENCE_OVERLAYS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
