#!/usr/bin/env python3
"""Visualize LabelMe GT vs full-inference YOLO overlap for all Raw_Patches images."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.pyplot import imread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.palm_analyzer import (
    PalmInstance,
    extract_palm_instances_in_annotation_order,
)

RAW_PATCHES_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "full_inference" / "predictions_full.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_gt_overlap_full"
SUMMARY_CSV = OUTPUT_DIR / "overlap_summary.csv"
CONTACT_SHEET_PATH = OUTPUT_DIR / "contact_sheet.png"

CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
MATCH_IOU_THRESHOLD = 0.5
PROGRESS_INTERVAL = 50
CONTACT_SHEET_COUNT = 25
CONTACT_SHEET_COLS = 5
SAVE_DPI = 100

COLOR_GT = "lime"
COLOR_YOLO = "red"
COLOR_SCORE = "yellow"
COLOR_MATCH = "cyan"
COLOR_TITLE = "white"


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    return [record for record in data if isinstance(record, dict)]


def group_predictions_by_image(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image_id = record.get("image_id")
        bbox = record.get("bbox")
        if image_id is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        score = record.get("score")
        grouped[str(image_id)].append(
            {
                "bbox": tuple(float(value) for value in bbox),
                "score": float(score) if score is not None else None,
            }
        )
    return dict(grouped)


def find_png_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Raw_Patches directory not found: {root}")
    return sorted(path for path in root.rglob("*.png") if path.is_file())


def resolve_json_path(png_path: Path) -> Path | None:
    candidates = [
        png_path.with_suffix(".json"),
        png_path.parent / f"{png_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(RAW_PATCHES_ROOT.rglob(f"{png_path.stem}.json"))
    return matches[0] if matches else None


def palm_bbox(palm: PalmInstance) -> tuple[float, float, float, float]:
    return palm.bbox_x, palm.bbox_y, palm.bbox_width, palm.bbox_height


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


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def filter_by_confidence(
    predictions: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
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


def compute_overlap_metrics(
    gt_bboxes: list[tuple[float, float, float, float]],
    yolo_bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, int]:
    """Return max pairwise IoU, mean best IoU per GT, and GT matches at IoU >= 0.5."""
    if not gt_bboxes or not yolo_bboxes:
        return 0.0, 0.0, 0

    best_ious: list[float] = []
    global_max = 0.0

    for gt_bbox in gt_bboxes:
        gt_best = 0.0
        for yolo_bbox in yolo_bboxes:
            overlap = iou_xywh(gt_bbox, yolo_bbox)
            global_max = max(global_max, overlap)
            gt_best = max(gt_best, overlap)
        best_ious.append(gt_best)

    mean_best = sum(best_ious) / len(best_ious)
    num_matches = sum(best >= MATCH_IOU_THRESHOLD for best in best_ious)
    return global_max, mean_best, num_matches


def draw_overlap_image(
    image: np.ndarray,
    image_id: str,
    gt_bboxes: list[tuple[float, float, float, float]],
    yolo_predictions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    height, width = image.shape[:2]

    fig = plt.figure(figsize=(width / SAVE_DPI, height / SAVE_DPI), dpi=SAVE_DPI)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for gt_bbox in gt_bboxes:
        x, y, w, h = gt_bbox
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=3,
                edgecolor=COLOR_GT,
                facecolor="none",
            )
        )

    for prediction in yolo_predictions:
        x, y, w, h = prediction["bbox"]
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor=COLOR_YOLO,
                facecolor="none",
            )
        )

        score = prediction.get("score")
        label = f"{score:.2f}" if score is not None else "n/a"
        ax.text(
            x,
            max(y - 4, 2),
            label,
            color=COLOR_SCORE,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    for gt_bbox in gt_bboxes:
        gt_center = bbox_center(gt_bbox)
        for prediction in yolo_predictions:
            yolo_bbox = prediction["bbox"]
            if iou_xywh(gt_bbox, yolo_bbox) <= 0.0:
                continue
            yolo_center = bbox_center(yolo_bbox)
            ax.plot(
                [gt_center[0], yolo_center[0]],
                [gt_center[1], yolo_center[1]],
                color=COLOR_MATCH,
                linewidth=0.8,
                alpha=0.8,
            )

    title = (
        f"{image_id} | GT: {len(gt_bboxes)} | "
        f"YOLO (conf>={CONFIDENCE_THRESHOLD}, NMS): {len(yolo_predictions)}"
    )
    ax.text(
        8,
        20,
        title,
        color=COLOR_TITLE,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, pad_inches=0)
    plt.close(fig)


def build_contact_sheet(overlay_paths: list[Path], output_path: Path) -> None:
    if not overlay_paths:
        print("No overlays available for contact sheet.")
        return

    selected = overlay_paths[:CONTACT_SHEET_COUNT]
    rows = CONTACT_SHEET_COLS
    cols = CONTACT_SHEET_COLS

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes_flat = np.atleast_1d(axes).flatten()

    for axis in axes_flat:
        axis.axis("off")

    for axis, overlay_path in zip(axes_flat, selected):
        image = imread(str(overlay_path))
        axis.imshow(image)
        axis.set_title(overlay_path.stem.replace("_overlap", ""), fontsize=8)
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"Prediction file not found: {PREDICTIONS_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_predictions = load_predictions(PREDICTIONS_PATH)
    predictions_by_image = group_predictions_by_image(all_predictions)
    png_paths = find_png_images(RAW_PATCHES_ROOT)

    print("YOLO vs GT overlap visualization (full inference)")
    print(f"  Raw_Patches: {RAW_PATCHES_ROOT}")
    print(f"  Predictions: {PREDICTIONS_PATH}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Confidence:  >= {CONFIDENCE_THRESHOLD}")
    print(f"  NMS IoU:     {NMS_IOU_THRESHOLD}")
    print(f"  PNG count:   {len(png_paths)}")
    print()

    summary_rows: list[dict[str, Any]] = []
    saved_overlay_paths: list[Path] = []

    total_gt = 0
    total_raw_yolo = 0
    total_filtered_yolo = 0
    total_nms_yolo = 0
    total_matches = 0

    for index, png_path in enumerate(png_paths, start=1):
        image_id = png_path.stem
        raw_predictions = predictions_by_image.get(image_id, [])
        raw_count = len(raw_predictions)

        filtered_predictions = filter_by_confidence(raw_predictions, CONFIDENCE_THRESHOLD)
        filtered_count = len(filtered_predictions)

        nms_predictions = apply_nms(filtered_predictions, NMS_IOU_THRESHOLD)
        nms_count = len(nms_predictions)

        json_path = resolve_json_path(png_path)
        if json_path is not None:
            palms = extract_palm_instances_in_annotation_order(json_path)
            gt_bboxes = [palm_bbox(palm) for palm in palms]
        else:
            gt_bboxes = []

        gt_count = len(gt_bboxes)
        yolo_bboxes = [prediction["bbox"] for prediction in nms_predictions]
        max_iou, mean_best_iou, num_matches = compute_overlap_metrics(gt_bboxes, yolo_bboxes)

        image = imread(str(png_path))
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        overlay_path = OUTPUT_DIR / f"{image_id}_overlap.png"
        draw_overlap_image(image, image_id, gt_bboxes, nms_predictions, overlay_path)
        saved_overlay_paths.append(overlay_path)

        summary_rows.append(
            {
                "image_id": image_id,
                "gt_count": gt_count,
                "raw_yolo_count": raw_count,
                "filtered_yolo_count": filtered_count,
                "nms_yolo_count": nms_count,
                "max_iou": round(max_iou, 4),
                "mean_best_iou": round(mean_best_iou, 4),
                "num_matches_iou_05": num_matches,
            }
        )

        total_gt += gt_count
        total_raw_yolo += raw_count
        total_filtered_yolo += filtered_count
        total_nms_yolo += nms_count
        total_matches += num_matches

        if index % PROGRESS_INTERVAL == 0 or index == len(png_paths):
            print(f"Processed {index}/{len(png_paths)} images")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    build_contact_sheet(saved_overlay_paths, CONTACT_SHEET_PATH)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"total images: {len(png_paths)}")
    print(f"total GT boxes: {total_gt}")
    print(f"total YOLO boxes before filtering: {total_raw_yolo}")
    print(f"total YOLO boxes after filtering: {total_filtered_yolo}")
    print(f"total YOLO boxes after NMS: {total_nms_yolo}")
    print(f"total IoU>=0.5 matches: {total_matches}")
    print()
    print(f"Saved summary CSV: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved contact sheet: {CONTACT_SHEET_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
