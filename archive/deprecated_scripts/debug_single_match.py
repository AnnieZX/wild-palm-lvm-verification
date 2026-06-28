#!/usr/bin/env python3
"""Debug GT vs YOLO bbox alignment for a single Raw_Patches image."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.pyplot import imread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.palm_analyzer import extract_palm_instances_in_annotation_order

PREDICTIONS_PATH = Path(
    "/deac/csc/yangGrp/cuij/palm/testing/results/yolo_new/"
    "yolo11x_new_datanew-yolo/val/predictions.json"
)
RAW_PATCHES_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug_single_match"

IMAGE_NAME = "100_0003_0001_7"
SAVE_DPI = 100


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


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return bbox[2] * bbox[3]


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    x, y, w, h = bbox
    return f"[x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}]"


def bbox_fully_inside_image(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    tolerance: float = 1.0,
) -> bool:
    x, y, w, h = bbox
    return (
        x >= -tolerance
        and y >= -tolerance
        and x + w <= width + tolerance
        and y + h <= height + tolerance
    )


def bbox_partially_inside_image(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> bool:
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    return not (x2 < 0 or y2 < 0 or x > width or y > height)


def resolve_raw_patches_file(stem: str, suffix: str) -> Path:
    direct = RAW_PATCHES_ROOT / f"{stem}{suffix}"
    if direct.exists():
        return direct

    matches = sorted(RAW_PATCHES_ROOT.rglob(f"{stem}{suffix}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find {stem}{suffix} under {RAW_PATCHES_ROOT}")


def load_predictions_for_image(
    records: list[dict[str, Any]],
    image_name: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        if extract_image_id(record) != image_name:
            continue
        bbox = extract_bbox(record)
        if bbox is None:
            continue
        predictions.append({"bbox": bbox, "score": extract_score(record)})
    return predictions


def print_gt_summary(gt_records: list[dict[str, Any]]) -> None:
    print("Ground-truth palms")
    for index, gt in enumerate(gt_records, start=1):
        center = gt["center"]
        center_text = (
            f"({center[0]:.2f}, {center[1]:.2f})" if center is not None else "none"
        )
        print(f"  GT{index}")
        print(f"    group_id: {gt['group_id']}")
        print(f"    bbox: {format_bbox(gt['bbox'])}")
        print(f"    center: {center_text}")
        print(f"    endpoint_count: {gt['endpoint_count']}")
    print()


def print_pairwise_iou(
    gt_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[float]:
    all_ious: list[float] = []

    for gt_index, gt in enumerate(gt_records, start=1):
        gt_bbox = gt["bbox"]
        print(f"GT{gt_index}")
        print(f"  bbox: {format_bbox(gt_bbox)}")

        best_iou = 0.0
        best_score: float | None = None

        for pred_index, prediction in enumerate(predictions, start=1):
            pred_bbox = prediction["bbox"]
            overlap = iou_xywh(gt_bbox, pred_bbox)
            all_ious.append(overlap)

            if overlap > best_iou:
                best_iou = overlap
                best_score = prediction["score"]

            print(f"  Prediction {pred_index}")
            print(f"    bbox: {format_bbox(pred_bbox)}")
            print(f"    IoU: {overlap:.4f}")

        score_text = f"{best_score:.3f}" if best_score is not None else "n/a"
        print(f"  Best IoU: {best_iou:.4f}")
        print(f"  Best prediction score: {score_text}")
        print()

    return all_ious


def print_scale_diagnostics(
    gt_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    width: int,
    height: int,
) -> None:
    print("Image information")
    print(f"  width: {width}")
    print(f"  height: {height}")
    print()

    gt_bboxes = [gt["bbox"] for gt in gt_records]
    pred_bboxes = [pred["bbox"] for pred in predictions]

    if gt_bboxes:
        largest_gt = max(gt_bboxes, key=bbox_area)
        avg_gt_w = sum(b[2] for b in gt_bboxes) / len(gt_bboxes)
        avg_gt_h = sum(b[3] for b in gt_bboxes) / len(gt_bboxes)
        print(f"  largest GT bbox: {format_bbox(largest_gt)} (area={bbox_area(largest_gt):.2f})")
        print(f"  average GT bbox size: w={avg_gt_w:.2f}, h={avg_gt_h:.2f}")
    else:
        print("  largest GT bbox: n/a")
        print("  average GT bbox size: n/a")

    if pred_bboxes:
        largest_pred = max(pred_bboxes, key=bbox_area)
        avg_pred_w = sum(b[2] for b in pred_bboxes) / len(pred_bboxes)
        avg_pred_h = sum(b[3] for b in pred_bboxes) / len(pred_bboxes)
        print(
            f"  largest prediction bbox: {format_bbox(largest_pred)} "
            f"(area={bbox_area(largest_pred):.2f})"
        )
        print(f"  average prediction bbox size: w={avg_pred_w:.2f}, h={avg_pred_h:.2f}")
    else:
        print("  largest prediction bbox: n/a")
        print("  average prediction bbox size: n/a")
    print()


def print_boundary_diagnostics(
    predictions: list[dict[str, Any]],
    width: int,
    height: int,
) -> None:
    if not predictions:
        print("Prediction boundary check: no predictions to inspect.")
        return

    fully_inside = 0
    partially_inside = 0
    fully_outside = 0

    for prediction in predictions:
        bbox = prediction["bbox"]
        if bbox_fully_inside_image(bbox, width, height):
            fully_inside += 1
        elif bbox_partially_inside_image(bbox, width, height):
            partially_inside += 1
        else:
            fully_outside += 1

    total = len(predictions)
    print("Prediction boundary check")
    print(f"  fully inside [{0}, {width}] x [{0}, {height}]: {fully_inside}/{total}")
    print(f"  partially inside: {partially_inside}/{total}")
    print(f"  fully outside: {fully_outside}/{total}")

    if fully_outside == total:
        print("  => Predictions appear OUTSIDE the image coordinate system.")
    elif fully_inside == total:
        print("  => All prediction boxes lie inside image boundaries.")
    else:
        print("  => Mixed boundary behavior; inspect visualization for alignment.")
    print()


def draw_debug_image(
    image: np.ndarray,
    gt_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
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

    for index, gt in enumerate(gt_records, start=1):
        x, y, w, h = gt["bbox"]
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=3,
                edgecolor="green",
                facecolor="none",
            )
        )
        ax.text(
            x,
            max(y - 4, 2),
            f"GT{index}",
            color="green",
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    for prediction in predictions:
        x, y, w, h = prediction["bbox"]
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
        score = prediction["score"]
        score_text = f"score={score:.3f}" if score is not None else "score=n/a"
        ax.text(
            x,
            max(y - 4, 2),
            score_text,
            color="red",
            fontsize=9,
            va="bottom",
            ha="left",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, pad_inches=0)
    plt.close(fig)


def main() -> None:
    image_path = resolve_raw_patches_file(IMAGE_NAME, ".png")
    json_path = resolve_raw_patches_file(IMAGE_NAME, ".json")
    output_path = OUTPUT_DIR / f"{IMAGE_NAME}_debug.png"

    if not PREDICTIONS_PATH.exists():
        print(f"Prediction file not found: {PREDICTIONS_PATH}")
        sys.exit(1)

    image = imread(str(image_path))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    height, width = image.shape[:2]

    palms = extract_palm_instances_in_annotation_order(json_path)
    gt_records = [
        {
            "group_id": palm.group_id,
            "bbox": (palm.bbox_x, palm.bbox_y, palm.bbox_width, palm.bbox_height),
            "center": (
                (palm.center_x, palm.center_y)
                if palm.center_x is not None and palm.center_y is not None
                else None
            ),
            "endpoint_count": palm.num_endpoints,
        }
        for palm in palms
    ]

    all_records = load_predictions(PREDICTIONS_PATH)
    predictions = load_predictions_for_image(all_records, IMAGE_NAME)

    print("Single-image GT vs YOLO debug")
    print(f"  image: {image_path}")
    print(f"  json:  {json_path}")
    print(f"  predictions: {PREDICTIONS_PATH}")
    print()

    print_gt_summary(gt_records)
    all_ious = print_pairwise_iou(gt_records, predictions)
    print_scale_diagnostics(gt_records, predictions, width, height)

    draw_debug_image(image, gt_records, predictions, output_path)
    print(f"Saved visualization: {output_path.relative_to(PROJECT_ROOT)}")
    print()

    total_gt = len(gt_records)
    total_pred = len(predictions)

    if all_ious:
        avg_iou = sum(all_ious) / len(all_ious)
        max_iou = max(all_ious)
        min_iou = min(all_ious)
    else:
        avg_iou = max_iou = min_iou = 0.0

    print("Summary")
    print(f"  Total GT palms: {total_gt}")
    print(f"  Total predictions: {total_pred}")
    print(f"  Average IoU (all GT x pred pairs): {avg_iou:.4f}")
    print(f"  Maximum IoU: {max_iou:.4f}")
    print(f"  Minimum IoU: {min_iou:.4f}")
    print()

    print_boundary_diagnostics(predictions, width, height)


if __name__ == "__main__":
    main()
