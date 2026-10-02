#!/usr/bin/env python3
"""
Purpose:
    Export a blind manual-audit set for the current Evaluation Protocol v2
    GT-negative (unmatched) verification detections.

    GT negative currently means "YOLO detection (conf >= 0.5) with no LabelMe
    palm box at IoU >= 0.5 under greedy one-to-one matching". It does NOT mean
    verified non-palm. This tool renders every such detection for human review
    and writes a manifest with empty manual-label columns. It never changes GT
    labels, thresholds or evaluation outputs, and never reads VLM predictions.

Input:
    - outputs/verification_dataset/index.csv
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON + PNG under Raw_Patches
    - outputs/evaluation_protocol_v2/**/A*_evaluation.csv (sample_id and
      matched_gt columns only, for a consistency cross-check)

Output (default outputs/diagnostics/gt_negative_audit/):
    - images/NNNN_sample_XXXXXX.png
    - manifest.csv
    - priority_review.csv
    - README.md

Example:
    python scripts/diagnostics/export_gt_negative_audit.py
    python scripts/diagnostics/export_gt_negative_audit.py --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_eval_spec = importlib.util.spec_from_file_location(
    "evaluate_verification_against_groundtruth",
    PROJECT_ROOT / "scripts" / "evaluate_verification_against_groundtruth.py",
)
_eval_module = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
_eval_spec.loader.exec_module(_eval_module)
compute_greedy_matches_for_index = _eval_module.compute_greedy_matches_for_index
IOU_THRESHOLD = _eval_module.IOU_THRESHOLD

from src.paths import (  # noqa: E402
    CURRENT_EVALUATION_ROOT,
    OUTPUTS_DIR,
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.preprocessing.gt_palm_bboxes import EVALUATION_PROTOCOL_VERSION  # noqa: E402
from src.preprocessing.json_parser import load_json  # noqa: E402
from src.preprocessing.verification_dataset import resolve_patch_image_from_index, build_png_index  # noqa: E402
from src.utils.labelme_paths import resolve_labelme_json  # noqa: E402
from src.utils.verification_index import find_yolo_prediction, index_bbox_from_row  # noqa: E402
from src.visualization.verification_visualization import _draw_text_with_outline, xywh_to_xyxy  # noqa: E402
from src.yolo.predictions_io import group_predictions_by_image, iou_xywh, load_predictions  # noqa: E402

DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "diagnostics" / "gt_negative_audit"
ALLOWED_OUTPUT_PARENT = OUTPUTS_DIR / "diagnostics"

EXPECTED_TOTAL = 5747
EXPECTED_MATCHED = 5109
EXPECTED_UNMATCHED = 638

BORDER_MARGIN_PX = 2.0
CROP_CONTEXT_FACTOR = 2.0
CROP_MIN_SIDE_PX = 128.0
CROP_TILE_PX = 456
PANEL_TEXT_HEIGHT_PX = 250
PNG_COMPRESSION = 6

# BGR colors
COLOR_TARGET = (0, 0, 255)
COLOR_GT_MATCHED_OTHER = (60, 200, 70)
COLOR_GT_FREE = (0, 220, 255)
COLOR_NEAREST_GT = (255, 0, 255)
COLOR_OTHER_DET = (200, 200, 200)
COLOR_CENTER = (255, 255, 0)
COLOR_PANEL = (28, 28, 28)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_DIM = (170, 170, 170)

MANUAL_COLUMNS = (
    "semantic",
    "cause",
    "truncated_at_border",
    "box_quality",
    "auditor_confidence",
    "notes",
)

MANUAL_VOCABULARY = {
    "semantic": ("palm", "non_palm", "ambiguous"),
    "cause": (
        "annotation_missing",
        "duplicate_detection",
        "localization_mismatch",
        "matching_artifact",
        "false_positive",
        "other",
    ),
    "truncated_at_border": ("true", "false"),
    "box_quality": ("good", "partial_crown", "oversized_multi", "undersized", "not_applicable"),
    "auditor_confidence": ("high", "low"),
    "notes": ("free text",),
}

METADATA_COLUMNS = (
    "audit_index",
    "sample_id",
    "image_id",
    "image_filename",
    "image_path",
    "yolo_confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "relative_bbox_area",
    "nearest_gt_iou",
    "nearest_gt_index",
    "nearest_gt_bbox",
    "nearest_gt_matched_to_sample",
    "num_gt_palms",
    "num_yolo_detections_ge_05",
    "touches_border",
    "nearest_other_detection_iou",
    "nearest_other_detection_sample_id",
    "nearest_other_detection_matched_gt",
    "center_points_inside_target",
    "visualization_path",
)

PRIORITY_COLUMNS = (
    "priority_rank",
    "priority_score",
    "priority_reasons",
    "audit_index",
    "sample_id",
    "image_id",
    "yolo_confidence",
    "nearest_gt_iou",
    "nearest_other_detection_iou",
    "touches_border",
    "center_points_inside_target",
    "visualization_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a blind manual-audit set for current Protocol v2 GT-negative detections.",
    )
    parser.add_argument("--index-csv", type=Path, default=VERIFICATION_DATASET_INDEX_CSV)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_FULL_JSON)
    parser.add_argument("--annotations-root", type=Path, default=RAW_PATCHES_ROOT)
    parser.add_argument("--images-root", type=Path, default=RAW_PATCHES_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=CURRENT_EVALUATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate an existing audit directory (refused if manual labels were entered)",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"\nSTOP: {message}")
    print("No partial audit dataset was written.")
    sys.exit(1)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved = output_dir.resolve()
    allowed = ALLOWED_OUTPUT_PARENT.resolve()
    if allowed not in resolved.parents:
        fail(f"output dir must be inside {allowed}, got {resolved}")

    if not resolved.exists() or not any(resolved.iterdir()):
        return
    if not overwrite:
        fail(f"{resolved} already exists and is not empty; pass --overwrite to regenerate it")

    manifest_path = resolved / "manifest.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        present = [column for column in MANUAL_COLUMNS if column in existing.columns]
        if any((existing[column].str.strip() != "").any() for column in present):
            fail(f"{manifest_path} contains manual audit labels; refusing to overwrite it")

    shutil.rmtree(resolved)


def resolve_yolo_bbox_and_confidence(
    row: pd.Series,
    predictions_by_image: dict[str, list[dict[str, Any]]],
) -> tuple[tuple[float, float, float, float], float | None]:
    """Same bbox resolution as the evaluator (find_yolo_prediction, index bbox fallback)."""
    index_bbox = index_bbox_from_row(row)
    yolo_bbox, yolo_confidence = find_yolo_prediction(
        str(row["image_name"]),
        index_bbox,
        predictions_by_image,
    )
    if yolo_bbox is None:
        return index_bbox, None
    return yolo_bbox, yolo_confidence


def cross_check_stored_evaluations(
    evaluation_root: Path,
    matched_by_sample: dict[str, bool],
) -> int:
    """Compare matched_gt against every stored full-size evaluation CSV; return files checked."""
    checked = 0
    for csv_path in sorted(evaluation_root.rglob("A*_evaluation.csv")):
        stored = pd.read_csv(csv_path, usecols=["sample_id", "matched_gt"], dtype=str)
        if len(stored) != EXPECTED_TOTAL:
            continue
        stored_matched = stored["matched_gt"].str.strip().str.lower().eq("true")
        for sample_id, is_matched in zip(stored["sample_id"], stored_matched):
            if matched_by_sample.get(sample_id) != bool(is_matched):
                fail(f"matched_gt for {sample_id} differs from stored {csv_path}")
        checked += 1
    return checked


def load_center_points(json_path: Path | None) -> list[tuple[float, float]]:
    if json_path is None:
        return []
    centers: list[tuple[float, float]] = []
    for shape in load_json(json_path).get("shapes", []):
        if not isinstance(shape, dict) or shape.get("label") != "center":
            continue
        points = shape.get("points", [])
        if points and isinstance(points[0], (list, tuple)) and len(points[0]) >= 2:
            centers.append((float(points[0][0]), float(points[0][1])))
    return centers


def touches_border(bbox: tuple[float, float, float, float], width: int, height: int) -> bool:
    x, y, w, h = bbox
    return (
        x <= BORDER_MARGIN_PX
        or y <= BORDER_MARGIN_PX
        or x + w >= width - BORDER_MARGIN_PX
        or y + h >= height - BORDER_MARGIN_PX
    )


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y, w, h = bbox
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def context_crop_window(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) of a ~2x context window, shifted inside and clipped to the image."""
    x, y, w, h = bbox
    crop_w = min(float(width), max(CROP_MIN_SIDE_PX, w * CROP_CONTEXT_FACTOR))
    crop_h = min(float(height), max(CROP_MIN_SIDE_PX, h * CROP_CONTEXT_FACTOR))
    cx, cy = x + w / 2.0, y + h / 2.0
    x1 = min(max(0.0, cx - crop_w / 2.0), width - crop_w)
    y1 = min(max(0.0, cy - crop_h / 2.0), height - crop_h)
    x1_i, y1_i = int(np.floor(max(0.0, x1))), int(np.floor(max(0.0, y1)))
    x2_i = min(width, int(np.ceil(x1 + crop_w)))
    y2_i = min(height, int(np.ceil(y1 + crop_h)))
    return x1_i, y1_i, x2_i, y2_i


def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 10,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    for start, end in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(np.hypot(end[0] - start[0], end[1] - start[1]))
        for offset in range(0, max(length, 1), dash * 2):
            t0 = offset / max(length, 1)
            t1 = min(offset + dash, length) / max(length, 1)
            p0 = (int(round(start[0] + (end[0] - start[0]) * t0)), int(round(start[1] + (end[1] - start[1]) * t0)))
            p1 = (int(round(start[0] + (end[0] - start[0]) * t1)), int(round(start[1] + (end[1] - start[1]) * t1)))
            cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def draw_overlays(
    canvas: np.ndarray,
    record: dict[str, Any],
    *,
    transform: tuple[float, float, float] = (1.0, 0.0, 0.0),
    labels: bool = True,
) -> None:
    """Draw GT boxes, other detections, center points and the target onto canvas.

    transform = (scale, offset_x, offset_y) maps image pixels to canvas pixels.
    """
    scale, offset_x, offset_y = transform

    def to_canvas(bbox: tuple[float, float, float, float]) -> tuple[tuple[int, int], tuple[int, int]]:
        x1, y1, x2, y2 = xywh_to_xyxy(
            (bbox[0] * scale + offset_x, bbox[1] * scale + offset_y, bbox[2] * scale, bbox[3] * scale)
        )
        return (x1, y1), (x2, y2)

    font_scale = 0.45
    for other in record["other_detections"]:
        p1, p2 = to_canvas(other["bbox"])
        cv2.rectangle(canvas, p1, p2, COLOR_OTHER_DET, 1, cv2.LINE_AA)
        if labels:
            _draw_text_with_outline(canvas, f"{other['confidence']:.2f}", (p1[0] + 2, p2[1] - 4), font_scale * 0.85, 1)

    for gt_index, gt_bbox in enumerate(record["gt_bboxes"]):
        owner = record["gt_owner"].get(gt_index)
        color = COLOR_GT_MATCHED_OTHER if owner else COLOR_GT_FREE
        p1, p2 = to_canvas(gt_bbox)
        cv2.rectangle(canvas, p1, p2, color, 2, cv2.LINE_AA)
        if labels:
            tag = f"GT{gt_index}" if owner else f"GT{gt_index} (no det match)"
            _draw_text_with_outline(canvas, tag, (p1[0] + 2, p1[1] + 14), font_scale, 1)

    nearest_index = record["nearest_gt_index"]
    if nearest_index is not None:
        p1, p2 = to_canvas(record["gt_bboxes"][nearest_index])
        draw_dashed_rectangle(canvas, p1, p2, COLOR_NEAREST_GT, 3)
        if labels:
            _draw_text_with_outline(
                canvas,
                f"nearest GT{nearest_index} IoU={record['nearest_gt_iou']:.3f} NOT matched to target",
                (p1[0], min(canvas.shape[0] - 6, p2[1] + 16)),
                font_scale,
                1,
            )

    for cx, cy in record["center_points"]:
        center = (int(round(cx * scale + offset_x)), int(round(cy * scale + offset_y)))
        cv2.circle(canvas, center, 5, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 3, COLOR_CENTER, -1, cv2.LINE_AA)

    p1, p2 = to_canvas(record["bbox"])
    cv2.rectangle(canvas, p1, p2, COLOR_TARGET, 3, cv2.LINE_AA)
    if labels:
        _draw_text_with_outline(
            canvas,
            f"TARGET {record['confidence']:.3f} UNMATCHED",
            (p1[0], max(14, p1[1] - 6)),
            font_scale * 1.1,
            1,
        )


def render_crop_tile(image: np.ndarray, record: dict[str, Any], annotated: bool) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = context_crop_window(record["bbox"], width, height)
    crop = image[y1:y2, x1:x2]
    scale = min(CROP_TILE_PX / crop.shape[1], CROP_TILE_PX / crop.shape[0])
    new_w, new_h = max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    tile = np.full((CROP_TILE_PX, CROP_TILE_PX, 3), COLOR_PANEL, dtype=np.uint8)
    pad_x, pad_y = (CROP_TILE_PX - new_w) // 2, (CROP_TILE_PX - new_h) // 2
    tile[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    if annotated:
        region = tile[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
        draw_overlays(region, record, transform=(scale, -x1 * scale, -y1 * scale), labels=False)
    caption = f"context crop x{scale:.1f} ({'annotated' if annotated else 'raw pixels'})"
    _draw_text_with_outline(tile, caption, (8, 18), 0.5, 1)
    return tile


def nearest_gt_description(record: dict[str, Any]) -> str:
    index = record["nearest_gt_index"]
    if index is None:
        return "Nearest GT: none (IoU = 0 with every LabelMe palm box)"
    owner = record["gt_owner"].get(index)
    if owner is None:
        status = "not matched to any conf>=0.5 detection"
    else:
        status = f"matched to {owner}"
    return f"Nearest GT: GT{index}  IoU={record['nearest_gt_iou']:.3f}  ({status}; NOT matched to target)"


def render_audit_panel(image: np.ndarray, record: dict[str, Any], audit_index: int, total: int) -> np.ndarray:
    full = image.copy()
    draw_overlays(full, record)
    right = np.vstack([render_crop_tile(image, record, True), render_crop_tile(image, record, False)])
    if right.shape[0] != full.shape[0]:
        right = cv2.resize(right, (int(round(right.shape[1] * full.shape[0] / right.shape[0])), full.shape[0]))
    top = np.hstack([full, right])

    panel = np.full((PANEL_TEXT_HEIGHT_PX, top.shape[1], 3), COLOR_PANEL, dtype=np.uint8)
    x, y, w, h = record["bbox"]
    if record["nearest_other_sample_id"]:
        other_text = (
            f"{record['nearest_other_iou']:.3f} ({record['nearest_other_sample_id']}, "
            f"{'GT+' if record['nearest_other_matched'] else 'GT-'})"
        )
    elif record["other_detections"]:
        other_text = "0.000 (no overlapping detection)"
    else:
        other_text = "n/a (only detection in image)"
    lines = [
        (f"GT-NEGATIVE AUDIT {audit_index:04d}/{total:04d}   {record['sample_id']}   image {record['image_filename']}", COLOR_TEXT),
        ("Protocol v2 status: UNMATCHED detection (no LabelMe palm box at IoU >= 0.5 under greedy one-to-one). NOT verified non-palm.", COLOR_TEXT_DIM),
        (f"YOLO confidence {record['confidence']:.4f}   bbox [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]   "
         f"area {w * h:.0f} px^2 ({100 * record['relative_area']:.2f}% of image)   touches border: {'YES' if record['touches_border'] else 'no'}", COLOR_TEXT),
        (nearest_gt_description(record), COLOR_TEXT),
        (f"GT palms in image: {len(record['gt_bboxes'])}   YOLO detections >= 0.5 in image: {record['num_detections']}   "
         f"LabelMe center points inside target: {record['centers_inside']}", COLOR_TEXT),
        (f"Nearest other detection IoU: {other_text}", COLOR_TEXT),
    ]
    text_y = 26
    for text, color in lines:
        cv2.putText(panel, text, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        text_y += 30

    legend = [
        (COLOR_TARGET, "target (unmatched)"),
        (COLOR_NEAREST_GT, "nearest GT (dashed, not matched)"),
        (COLOR_GT_MATCHED_OTHER, "GT matched to another det"),
        (COLOR_GT_FREE, "GT with no det match"),
        (COLOR_OTHER_DET, "other det >= 0.5"),
        (COLOR_CENTER, "LabelMe center point"),
    ]
    legend_x = 12
    legend_y = PANEL_TEXT_HEIGHT_PX - 22
    for color, label in legend:
        cv2.rectangle(panel, (legend_x, legend_y - 10), (legend_x + 22, legend_y + 2), color, -1)
        cv2.putText(panel, label, (legend_x + 28, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        legend_x += 36 + int(len(label) * 8.2)
    return np.vstack([top, panel])


def priority_for(row: dict[str, Any]) -> tuple[int, str]:
    """Model-independent review-order heuristic; never used as a label."""
    score = 0
    reasons: list[str] = []
    confidence = row["yolo_confidence"]
    if confidence >= 0.7:
        score += 2
        reasons.append("conf>=0.7")
    elif confidence >= 0.6:
        score += 1
        reasons.append("conf>=0.6")
    iou = row["nearest_gt_iou"]
    if iou == 0.0:
        score += 2
        reasons.append("isolated_iou0")
    elif iou >= IOU_THRESHOLD:
        score += 2
        reasons.append("iou>=0.5_gt_taken")
    elif iou >= 0.4:
        score += 2
        reasons.append("iou_near_threshold")
    if row["nearest_other_detection_iou"] != "" and float(row["nearest_other_detection_iou"]) >= 0.5:
        score += 1
        reasons.append("duplicate_like_overlap")
    if row["center_points_inside_target"] > 0:
        score += 2
        reasons.append("center_point_inside")
    if row["touches_border"] == "true":
        score += 1
        reasons.append("edge")
    return score, ";".join(reasons)


README_TEMPLATE = """# Manual GT-negative audit (Evaluation Protocol {protocol})

Generated by `scripts/diagnostics/export_gt_negative_audit.py`. This directory is a
diagnostic artifact. It does **not** modify official evaluation labels, thresholds,
prompts, predictions or any file outside this directory.

## Why this audit exists

Under the current protocol a verification detection is GT negative when it is a
YOLO detection with confidence >= 0.5 that has **no LabelMe palm box at IoU >= {iou}**
under per-image greedy one-to-one matching. GT negative therefore means
**"unmatched detection"**, not verified **"non-palm"**. A read-only audit found that the
LabelMe annotations are not documented as exhaustive and contain palms without
boxes, so the unmatched set mixes true false positives, palms missing from LabelMe,
duplicate detections, localization near-misses, matching artifacts and ambiguous
cases. These must be separated by a human before the negatives are used for
fine-tuning or the protocol is changed.

## Contents

| File | Description |
|---|---|
| `manifest.csv` | All {n_neg} current Protocol v2 GT-negative detections, sorted by `sample_id`, with computed metadata and **empty** manual columns. This is the file to fill in. |
| `priority_review.csv` | The same {n_neg} detections in a suggested review order (model-independent heuristic). Review-order helper only; do not enter labels here. |
| `images/NNNN_sample_XXXXXX.png` | One audit panel per detection; `NNNN` = `audit_index` in `manifest.csv`. |

Population checks at generation time: {n_total} detections, {n_pos} matched (GT+),
{n_neg} unmatched (GT-); matched/unmatched cross-checked against {n_checked} stored
Protocol v2 evaluation CSVs. VLM predictions were not read; the audit is blind to them.

## Reading a panel

- Left: full 912x912 patch. Right: context crop (~2x the box, shifted inside and
  clipped at the image border), annotated on top and raw pixels below.
- **Red, thick**: the target detection (always unmatched in this audit).
- **Magenta, dashed**: the nearest LabelMe palm box by IoU. It is **not** matched to
  the target. If its IoU is >= 0.5 it was taken by another detection with higher IoU.
- **Green**: LabelMe palm box matched to another detection.
- **Yellow**: LabelMe palm box not matched to any conf >= 0.5 detection.
- **Light gray, thin**: other YOLO detections with conf >= 0.5 (confidence printed).
- **Cyan dots**: LabelMe `center` points.
- The text panel lists sample id, image, confidence, bbox, area, border contact,
  nearest GT (index, IoU, who it is matched to), GT and detection counts, and the
  nearest other detection.

## Computed manifest columns (do not edit)

- `nearest_gt_iou` / `nearest_gt_index` / `nearest_gt_bbox`: highest-IoU LabelMe palm box
  ignoring assignment; index and bbox are blank when the IoU is 0 with every box.
- `nearest_gt_matched_to_sample`: the detection that owns that GT box under the official
  matching; blank if no conf >= 0.5 detection matched it.
- `nearest_other_detection_iou`: highest IoU with another conf >= 0.5 detection in the same
  image (blank if the target is the only one). The `_sample_id` / `_matched_gt` columns are
  blank when that IoU is 0.
- `touches_border`: box edge within {border_px} px of the patch edge.
- `center_points_inside_target`: LabelMe `center` points inside the target box.
- `image_path`: raw patch PNG; `visualization_path`: audit panel, relative to the repository root.

## Annotation schema (fill in `manifest.csv`)

| Column | Allowed values | Meaning |
|---|---|---|
| `semantic` (required) | `palm`, `non_palm`, `ambiguous` | What the target box actually shows. |
| `cause` (required) | `annotation_missing`, `duplicate_detection`, `localization_mismatch`, `matching_artifact`, `false_positive`, `other` | Why it is unmatched. |
| `truncated_at_border` | `true`, `false` | The palm/object is cut off by the patch edge. |
| `box_quality` | `good`, `partial_crown`, `oversized_multi`, `undersized`, `not_applicable` | How well the box fits the object (use `not_applicable` for `non_palm`). |
| `auditor_confidence` | `high`, `low` | Your confidence in this row. |
| `notes` | free text | Anything unusual (e.g. LabelMe label typo, point-only annotation). |

Guidance for `cause`:

- `annotation_missing`: a real palm with no LabelMe box for it.
- `duplicate_detection`: a real palm already covered by another detection that is matched.
- `localization_mismatch`: a real palm that has a LabelMe box, but the YOLO box fits it poorly (IoU < 0.5).
- `matching_artifact`: unmatched because of the matching/labels themselves (e.g. palm box mislabeled, point-only palm, assignment order).
- `false_positive`: not a palm (vegetation, shadow, other tree, background).
- `other`: none of the above; explain in `notes`.

Typical consistent combinations: `palm` + {{`annotation_missing`, `duplicate_detection`,
`localization_mismatch`, `matching_artifact`}}; `non_palm` + `false_positive`;
`ambiguous` + any cause with `auditor_confidence = low`.

## How to fill out manifest.csv

1. Keep every row and every computed column unchanged; edit only the six manual columns.
2. Use exactly the lowercase values above; leave a cell empty only if not yet reviewed.
3. Open `images/<audit_index>_<sample_id>.png` via the `visualization_path` column.
4. Optionally follow `priority_review.csv` for order, but record labels in `manifest.csv` by `sample_id`.
5. Do not look at VLM outputs while labelling.
6. Save as CSV (UTF-8, comma-separated). Re-running the exporter refuses to overwrite a
   manifest that already contains manual labels.

## Priority heuristic (`priority_review.csv`)

Score = conf >= 0.7 (+2) or conf >= 0.6 (+1); nearest GT IoU == 0 (+2), IoU >= 0.5
with the GT taken (+2) or 0.4 <= IoU < 0.5 (+2); nearest other detection IoU >= 0.5 (+1);
LabelMe center point inside the target (+2); touches the border (+1). Sorted by score,
then confidence, then `sample_id`. It uses no VLM output and is not a label.

## What happens next

This audit does **not** change the official Protocol v2 labels (GT+ {n_pos} / GT- {n_neg}),
the IoU threshold, the YOLO confidence threshold or any evaluation output. Any protocol
change (e.g. relabelling or excluding negatives) will be decided only after the
completed audit has been reviewed, and would be introduced as a new, versioned protocol.
"""


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    images_dir = output_dir / "images"

    for path in (args.index_csv, args.predictions, args.annotations_root, args.images_root):
        if not path.exists():
            fail(f"required input not found: {path}")

    index_df = pd.read_csv(args.index_csv)
    predictions_by_image = group_predictions_by_image(load_predictions(args.predictions))
    gt_cache: dict[str, list[tuple[float, float, float, float]]] = {}
    greedy_matches = compute_greedy_matches_for_index(
        index_df,
        predictions_by_image,
        args.annotations_root,
        gt_cache,
        IOU_THRESHOLD,
    )

    total = len(index_df)
    matched_by_sample = {sample_id: match.matched_gt for sample_id, match in greedy_matches.items()}
    n_matched = sum(matched_by_sample.values())
    n_unmatched = total - n_matched
    print(f"Verification detections: {total}  matched (GT+): {n_matched}  unmatched (GT-): {n_unmatched}")
    if index_df["sample_id"].duplicated().any():
        fail("duplicate sample_id values in the verification index")
    if len(greedy_matches) != total:
        fail(f"matching returned {len(greedy_matches)} results for {total} detections")
    if (total, n_matched, n_unmatched) != (EXPECTED_TOTAL, EXPECTED_MATCHED, EXPECTED_UNMATCHED):
        fail(
            f"expected {EXPECTED_TOTAL}/{EXPECTED_MATCHED}/{EXPECTED_UNMATCHED}, "
            f"got {total}/{n_matched}/{n_unmatched}"
        )

    n_checked = cross_check_stored_evaluations(args.evaluation_root, matched_by_sample)
    if n_checked == 0:
        fail(f"no stored {EXPECTED_TOTAL}-row evaluation CSV found under {args.evaluation_root}")
    print(f"matched_gt identical to {n_checked} stored Protocol v2 evaluation CSVs")

    prepare_output_dir(output_dir, args.overwrite)
    images_dir.mkdir(parents=True, exist_ok=True)

    png_index = build_png_index(args.images_root)
    negative_ids = sorted(sample_id for sample_id, is_matched in matched_by_sample.items() if not is_matched)
    negative_set = set(negative_ids)
    rows: list[dict[str, Any]] = []
    audit_position = {sample_id: position for position, sample_id in enumerate(negative_ids, start=1)}
    confidence_lookup_mismatches = 0

    for image_name, group in index_df.groupby("image_name", sort=True):
        image_name = str(image_name)
        group_ids = [str(sample_id) for sample_id in group["sample_id"]]
        if not negative_set.intersection(group_ids):
            continue

        image_path = resolve_patch_image_from_index(args.images_root, image_name, png_index)
        image = cv2.imread(str(image_path)) if image_path is not None else None
        if image is None:
            fail(f"could not read image for {image_name}")
        height, width = image.shape[:2]

        json_path = resolve_labelme_json(args.annotations_root, image_name)
        gt_bboxes = gt_cache.get(image_name, [])
        center_points = load_center_points(json_path)

        detections: dict[str, dict[str, Any]] = {}
        for _, sample in group.iterrows():
            sample_id = str(sample["sample_id"])
            bbox, lookup_confidence = resolve_yolo_bbox_and_confidence(sample, predictions_by_image)
            confidence = float(sample["confidence"])
            if lookup_confidence is not None and abs(lookup_confidence - confidence) > 1e-6:
                confidence_lookup_mismatches += 1
            detections[sample_id] = {"bbox": bbox, "confidence": confidence}

        gt_owner = {
            greedy_matches[sample_id].gt_index: sample_id
            for sample_id in group_ids
            if greedy_matches[sample_id].matched_gt
        }

        for sample_id in group_ids:
            if sample_id not in negative_set:
                continue
            match = greedy_matches[sample_id]
            target = detections[sample_id]
            bbox = target["bbox"]
            others = [
                {"sample_id": other_id, **other}
                for other_id, other in detections.items()
                if other_id != sample_id
            ]
            nearest_other_iou, nearest_other_id = 0.0, ""
            for other in others:
                overlap = iou_xywh(bbox, other["bbox"])
                if overlap > nearest_other_iou:
                    nearest_other_iou, nearest_other_id = overlap, other["sample_id"]

            record = {
                "sample_id": sample_id,
                "image_filename": image_path.name,
                "bbox": bbox,
                "confidence": target["confidence"],
                "relative_area": bbox[2] * bbox[3] / float(width * height),
                "touches_border": touches_border(bbox, width, height),
                "gt_bboxes": gt_bboxes,
                "gt_owner": gt_owner,
                "nearest_gt_index": match.gt_index,
                "nearest_gt_iou": float(match.max_iou),
                "other_detections": others,
                "num_detections": len(detections),
                "center_points": center_points,
                "centers_inside": sum(point_in_bbox(point, bbox) for point in center_points),
                "nearest_other_iou": nearest_other_iou,
                "nearest_other_sample_id": nearest_other_id,
                "nearest_other_matched": bool(nearest_other_id) and matched_by_sample[nearest_other_id],
            }

            audit_index = audit_position[sample_id]
            file_name = f"{audit_index:04d}_{sample_id}.png"
            panel = render_audit_panel(image, record, audit_index, len(negative_ids))
            if not cv2.imwrite(str(images_dir / file_name), panel, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
                fail(f"failed to write {images_dir / file_name}")

            nearest_bbox = gt_bboxes[match.gt_index] if match.gt_index is not None else None
            rows.append(
                {
                    "audit_index": audit_index,
                    "sample_id": sample_id,
                    "image_id": image_name,
                    "image_filename": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "yolo_confidence": round(target["confidence"], 6),
                    "bbox_x": round(bbox[0], 3),
                    "bbox_y": round(bbox[1], 3),
                    "bbox_w": round(bbox[2], 3),
                    "bbox_h": round(bbox[3], 3),
                    "bbox_area": round(bbox[2] * bbox[3], 1),
                    "relative_bbox_area": round(record["relative_area"], 6),
                    "nearest_gt_iou": round(record["nearest_gt_iou"], 4),
                    "nearest_gt_index": "" if match.gt_index is None else int(match.gt_index),
                    "nearest_gt_bbox": ""
                    if nearest_bbox is None
                    else "[" + ", ".join(f"{value:.3f}" for value in nearest_bbox) + "]",
                    "nearest_gt_matched_to_sample": gt_owner.get(match.gt_index, "")
                    if match.gt_index is not None
                    else "",
                    "num_gt_palms": len(gt_bboxes),
                    "num_yolo_detections_ge_05": len(detections),
                    "touches_border": "true" if record["touches_border"] else "false",
                    "nearest_other_detection_iou": round(nearest_other_iou, 4) if others else "",
                    "nearest_other_detection_sample_id": nearest_other_id,
                    "nearest_other_detection_matched_gt": ""
                    if not nearest_other_id
                    else ("true" if record["nearest_other_matched"] else "false"),
                    "center_points_inside_target": record["centers_inside"],
                    "visualization_path": str((images_dir / file_name).resolve().relative_to(PROJECT_ROOT)),
                }
            )

    manifest = pd.DataFrame(rows).sort_values("audit_index").reset_index(drop=True)
    manifest = manifest[list(METADATA_COLUMNS)]
    for column in MANUAL_COLUMNS:
        manifest[column] = ""
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    priority_rows = []
    for row in manifest.to_dict("records"):
        score, reasons = priority_for(row)
        priority_rows.append({**row, "priority_score": score, "priority_reasons": reasons})
    priority = pd.DataFrame(priority_rows).sort_values(
        ["priority_score", "yolo_confidence", "sample_id"],
        ascending=[False, False, True],
    )
    priority["priority_rank"] = range(1, len(priority) + 1)
    priority[list(PRIORITY_COLUMNS)].to_csv(output_dir / "priority_review.csv", index=False)

    (output_dir / "README.md").write_text(
        README_TEMPLATE.format(
            protocol=EVALUATION_PROTOCOL_VERSION,
            iou=IOU_THRESHOLD,
            n_total=total,
            n_pos=n_matched,
            n_neg=n_unmatched,
            n_checked=n_checked,
            border_px=f"{BORDER_MARGIN_PX:g}",
        ),
        encoding="utf-8",
    )

    validate_outputs(output_dir, negative_ids, matched_by_sample)

    print(f"\nconfidence lookup mismatches (index vs predictions JSON): {confidence_lookup_mismatches}")
    print_summary(manifest)
    print(f"\nWrote {len(manifest)} panels, manifest.csv, priority_review.csv, README.md to {output_dir}")


def validate_outputs(output_dir: Path, negative_ids: list[str], matched_by_sample: dict[str, bool]) -> None:
    manifest = pd.read_csv(output_dir / "manifest.csv", dtype=str, keep_default_na=False)
    checks = {
        f"manifest has {EXPECTED_UNMATCHED} rows": len(manifest) == EXPECTED_UNMATCHED,
        f"{EXPECTED_UNMATCHED} panel files exist": len(list((output_dir / "images").glob("*.png"))) == EXPECTED_UNMATCHED,
        "sample_id values unique": manifest["sample_id"].is_unique,
        "every visualization_path exists": all((PROJECT_ROOT / path).is_file() for path in manifest["visualization_path"]),
        "no GT+ sample included": not any(matched_by_sample[sample_id] for sample_id in manifest["sample_id"]),
        "every GT- sample included exactly once": sorted(manifest["sample_id"]) == negative_ids,
        "manual columns all empty": all((manifest[column] == "").all() for column in MANUAL_COLUMNS),
        "priority_review covers the same samples": sorted(
            pd.read_csv(output_dir / "priority_review.csv", dtype=str)["sample_id"]
        ) == negative_ids,
    }
    print("\nValidation")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("\nValidation FAILED; do not use this audit directory.")
        sys.exit(1)


def print_summary(manifest: pd.DataFrame) -> None:
    confidence = manifest["yolo_confidence"].astype(float)
    iou = manifest["nearest_gt_iou"].astype(float)
    other = pd.to_numeric(manifest["nearest_other_detection_iou"], errors="coerce").fillna(0.0)
    print("\nDiagnostic summary (not labels)")
    print(f"  edge detections: {(manifest['touches_border'] == 'true').sum()}")
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print("  confidence: " + "  ".join(f"[{lo},{hi if hi <= 1 else 1.0}): {((confidence >= lo) & (confidence < hi)).sum()}" for lo, hi in bins))
    print(
        f"  nearest GT IoU: ==0: {(iou == 0).sum()}  (0,0.1): {((iou > 0) & (iou < 0.1)).sum()}  "
        f"[0.1,0.3): {((iou >= 0.1) & (iou < 0.3)).sum()}  [0.3,0.5): {((iou >= 0.3) & (iou < 0.5)).sum()}  "
        f">=0.5 (GT taken): {(iou >= 0.5).sum()}"
    )
    duplicate_like = (iou >= IOU_THRESHOLD) | (other >= 0.5)
    print(
        f"  duplicate-like (nearest GT IoU >= 0.5 but taken, or other detection IoU >= 0.5): {duplicate_like.sum()}"
        f"  [GT taken: {(iou >= IOU_THRESHOLD).sum()}, other det IoU >= 0.5: {(other >= 0.5).sum()}]"
    )
    print(f"  LabelMe center point inside target: {(manifest['center_points_inside_target'].astype(int) > 0).sum()}")


if __name__ == "__main__":
    main()
