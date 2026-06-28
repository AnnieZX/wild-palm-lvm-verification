#!/usr/bin/env python3
"""Run YOLO inference on all Raw_Patches PNGs and save predictions + overlays."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    "/deac/csc/yangGrp/cuij/palm/training/yolonew/results/yolo11x_palm_new/weights/best.pt"
)
IMAGES_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_inference"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
PREDICTIONS_JSON = OUTPUT_DIR / "predictions_full.json"

CONF_THRESHOLD = 0.001
MAX_DET = 300
PROGRESS_INTERVAL = 50

BOX_COLOR = (0, 255, 0)  # BGR green
TEXT_COLOR = (0, 255, 0)
TITLE_COLOR = (0, 255, 255)  # BGR yellow
BOX_THICKNESS = 2


def find_png_images(images_root: Path) -> list[Path]:
    """Return all PNG files under the images root, sorted by path."""
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    return sorted(path for path in images_root.rglob("*.png") if path.is_file())


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Convert corner box to COCO [x, y, width, height]."""
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def draw_overlay(
    image,
    image_name: str,
    detections: list[dict],
):
    """Draw green boxes, confidence labels, and image name on a copy of the image."""
    overlay = image.copy()

    title_y = 24
    cv2.putText(
        overlay,
        image_name,
        (10, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TITLE_COLOR,
        2,
        cv2.LINE_AA,
    )

    for detection in detections:
        x, y, width, height = detection["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOR, thickness=BOX_THICKNESS)

        score = detection["score"]
        label = f"{score:.3f}"
        text_x = max(x1, 0)
        text_y = max(y1 - 6, 20)
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


def run_inference_on_image(model, image_path: Path) -> list[dict]:
    """Run YOLO on one image and return COCO-format detections."""
    image_id = image_path.stem
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        max_det=MAX_DET,
        save=False,
        verbose=False,
    )

    detections: list[dict] = []
    if not results:
        return detections

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes_xyxy = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()

    for box, score, class_id in zip(boxes_xyxy, scores, classes):
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

    if not MODEL_PATH.exists():
        print(f"Model weights not found: {MODEL_PATH}")
        sys.exit(1)

    image_paths = find_png_images(IMAGES_ROOT)
    if not image_paths:
        print(f"No PNG images found under: {IMAGES_ROOT}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    print("Full YOLO inference on Raw_Patches")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Images:  {IMAGES_ROOT}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  conf:    {CONF_THRESHOLD}")
    print(f"  max_det: {MAX_DET}")
    print(f"  PNG count: {len(image_paths)}")
    print()

    model = YOLO(str(MODEL_PATH))

    all_predictions: list[dict] = []
    per_image_counts: list[int] = []
    all_scores: list[float] = []

    for index, image_path in enumerate(image_paths, start=1):
        image_name = image_path.stem
        detections = run_inference_on_image(model, image_path)
        all_predictions.extend(detections)

        per_image_counts.append(len(detections))
        all_scores.extend(detection["score"] for detection in detections)

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: could not read image for overlay: {image_path}")
        else:
            overlay = draw_overlay(image, image_name, detections)
            overlay_path = OVERLAY_DIR / f"{image_name}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

        if index % PROGRESS_INTERVAL == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} images")

    with PREDICTIONS_JSON.open("w", encoding="utf-8") as file:
        json.dump(all_predictions, file, indent=2)

    total_images = len(image_paths)
    total_detections = len(all_predictions)
    avg_detections = total_detections / total_images if total_images else 0.0
    avg_confidence = mean(all_scores) if all_scores else 0.0

    count_ge_05 = sum(score >= 0.5 for score in all_scores)
    count_ge_025 = sum(score >= 0.25 for score in all_scores)
    count_ge_01 = sum(score >= 0.1 for score in all_scores)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections/image: {avg_detections:.4f}")
    print(f"Average confidence: {avg_confidence:.6f}")
    print()
    print(f"score >= 0.5:  {count_ge_05}")
    print(f"score >= 0.25: {count_ge_025}")
    print(f"score >= 0.1:  {count_ge_01}")
    print()
    print(f"Saved predictions: {PREDICTIONS_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Saved overlays:    {OVERLAY_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
