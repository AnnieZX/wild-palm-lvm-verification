#!/usr/bin/env python3
"""
Purpose:
    Export two random, blind semantic-review samples that quantify how often the
    Evaluation Protocol v2 matching status disagrees with a human judgement of
    whether a YOLO detection visually shows a genuine palm:

    - PRIMARY: simple random sample (seed 42) of 100 of the 638 current GT-negative
      (matched_gt=False) detections, plus an optional SECONDARY cluster-aware
      sample of 100 for sensitivity analysis (manifest only).
    - QC: simple random sample of 100 of the 5,109 current GT-positive
      (matched_gt=True) detections, kept in a separate directory.

    Sampling uses only the sorted sample_id list of each population; no confidence,
    IoU, GT metadata, priority score or model output is used. Blind panels show
    only the raw patch, the target box and raw context crops with a neutral ID.
    Diagnostic metadata is written to a separate, sealed diagnostic_lookup.csv.

    This tool never changes GT labels, thresholds, matching, prompts, predictions or
    evaluation outputs, never reads VLM predictions and runs no inference.

Input:
    - outputs/verification_dataset/index.csv
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON + PNG under Raw_Patches
    - outputs/evaluation_protocol_v2/**/A*_evaluation.csv (sample_id, matched_gt only)
    - outputs/diagnostics/gt_negative_audit/manifest.csv (read-only cross-check)

Output:
    outputs/diagnostics/gt_negative_blind_pilot/
        README.md, SAMPLING_METHODS.md, blind_manifest.csv, diagnostic_lookup.csv,
        cluster_aware_manifest.csv, sampling_report.json, panels/blind_NNN.png
    outputs/diagnostics/gt_positive_blind_qc/
        README.md, SAMPLING_METHODS.md, blind_manifest.csv, diagnostic_lookup.csv,
        sampling_report.json, panels/qc_NNN.png

Example:
    python scripts/diagnostics/export_blind_semantic_pilot.py
    python scripts/diagnostics/export_blind_semantic_pilot.py --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_eval = _load_module(
    "evaluate_verification_against_groundtruth",
    PROJECT_ROOT / "scripts" / "evaluate_verification_against_groundtruth.py",
)
_audit = _load_module(
    "export_gt_negative_audit",
    PROJECT_ROOT / "scripts" / "diagnostics" / "export_gt_negative_audit.py",
)

from src.evaluation.gt_matching import best_gt_overlap  # noqa: E402
from src.paths import (  # noqa: E402
    CURRENT_EVALUATION_ROOT,
    OUTPUTS_DIR,
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.preprocessing.gt_palm_bboxes import EVALUATION_PROTOCOL_VERSION  # noqa: E402
from src.preprocessing.verification_dataset import build_png_index, resolve_patch_image_from_index  # noqa: E402
from src.utils.labelme_paths import resolve_labelme_json  # noqa: E402
from src.yolo.predictions_io import group_predictions_by_image, iou_xywh, load_predictions  # noqa: E402

IOU_THRESHOLD = _eval.IOU_THRESHOLD
EXPECTED_TOTAL = _audit.EXPECTED_TOTAL
EXPECTED_MATCHED = _audit.EXPECTED_MATCHED
EXPECTED_UNMATCHED = _audit.EXPECTED_UNMATCHED

DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
EXISTING_AUDIT_MANIFEST = DIAGNOSTICS_DIR / "gt_negative_audit" / "manifest.csv"
NEGATIVE_DIR = DIAGNOSTICS_DIR / "gt_negative_blind_pilot"
POSITIVE_DIR = DIAGNOSTICS_DIR / "gt_positive_blind_qc"

SAMPLE_SIZE = 100
NEGATIVE_SAMPLE_SEED = 42
NEGATIVE_REVIEW_ORDER_SEED = 2026
CLUSTER_AWARE_SEED = 42
POSITIVE_SAMPLE_SEED = 20260927
POSITIVE_REVIEW_ORDER_SEED = 2027

# Raw_Patches names are <flight>_<folder>_<frame>_<k>, k in 1..8; the eight patches of
# one parent frame tile a 2x4 grid (verified by edge continuity, see SAMPLING_METHODS.md).
PATCHES_PER_PARENT = 8

BLIND_MANIFEST_COLUMNS = (
    "blind_id",
    "sample_id",
    "blind_visualization_path",
    "semantic",
    "reviewer_confidence",
    "notes",
)
REVIEW_COLUMNS = ("semantic", "reviewer_confidence", "notes")
REVIEW_VOCABULARY = {
    "semantic": ("palm", "non_palm", "ambiguous"),
    "reviewer_confidence": ("high", "low"),
}
FORBIDDEN_MANIFEST_TOKENS = (
    "conf", "iou", "gt", "match", "nearest", "vlm", "priority", "cause", "decision",
    "reliable", "score", "border", "center", "label", "status",
)

LOOKUP_COLUMNS = (
    "blind_id",
    "review_position",
    "sample_id",
    "draw_position",
    "population",
    "audit_index",
    "image_id",
    "parent_id",
    "patch_position",
    "yolo_confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "relative_bbox_area",
    "protocol_v2_matched_gt",
    "protocol_v2_gt_index",
    "protocol_v2_iou",
    "nearest_gt_iou",
    "nearest_gt_index",
    "num_gt_palms",
    "num_yolo_detections_ge_05",
    "touches_border",
    "center_points_inside_target",
    "nearest_other_detection_iou",
    "nearest_other_detection_sample_id",
    "nearest_other_detection_matched_gt",
    "same_image_in_sample",
    "same_parent_in_sample",
)

PANEL_SIDE_PX = 912
TILE_PX = PANEL_SIDE_PX // 2
TILE_MARGIN_PX = 16
HEADER_PX = 44
CROP_NEAR = (2.0, 128.0)
CROP_WIDE = (4.0, 256.0)
PNG_COMPRESSION = 6

COLOR_TARGET = (0, 255, 255)
COLOR_BG = (28, 28, 28)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_DIM = (170, 170, 170)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export random blind semantic-review samples (no inference).")
    parser.add_argument("--index-csv", type=Path, default=VERIFICATION_DATASET_INDEX_CSV)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_FULL_JSON)
    parser.add_argument("--annotations-root", type=Path, default=RAW_PATCHES_ROOT)
    parser.add_argument("--images-root", type=Path, default=RAW_PATCHES_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=CURRENT_EVALUATION_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing pilot directories (refused if review labels were entered)",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"\nSTOP: {message}")
    sys.exit(1)


def parent_of(image_id: str) -> tuple[str, int]:
    stem, position = image_id.rsplit("_", 1)
    return stem, int(position)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved = output_dir.resolve()
    if DIAGNOSTICS_DIR.resolve() not in resolved.parents:
        fail(f"output dir must be inside {DIAGNOSTICS_DIR}, got {resolved}")
    if not resolved.exists() or not any(resolved.iterdir()):
        return
    if not overwrite:
        fail(f"{resolved} already exists and is not empty; pass --overwrite to regenerate it")
    manifest_path = resolved / "blind_manifest.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        if any((existing[column].str.strip() != "").any() for column in REVIEW_COLUMNS if column in existing):
            fail(f"{manifest_path} contains review labels; refusing to overwrite it")
    shutil.rmtree(resolved)


def simple_random_sample(population: list[str], size: int, seed: int) -> list[str]:
    """Draw from the sample_id-sorted population; the list order is the draw order."""
    return random.Random(seed).sample(sorted(population), size)


def shuffled_review_order(sample: list[str], seed: int) -> list[str]:
    order = sorted(sample)
    random.Random(seed).shuffle(order)
    return order


def cluster_aware_sample(
    population: list[str],
    image_of: dict[str, str],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """At most one detection per parent: random parents first, then one random detection each.

    If fewer than `size` parents exist, further rounds draw from parents with detections
    left, so a parent contributes a second detection only after every parent has one.
    """
    rng = random.Random(seed)
    by_parent: dict[str, list[str]] = {}
    for sample_id in sorted(population):
        by_parent.setdefault(parent_of(image_of[sample_id])[0], []).append(sample_id)
    remaining = {parent: list(ids) for parent, ids in by_parent.items()}
    chosen: list[dict[str, Any]] = []
    round_index = 0
    while len(chosen) < size:
        round_index += 1
        parents = sorted(parent for parent, ids in remaining.items() if ids)
        if not parents:
            break
        rng.shuffle(parents)
        for parent in parents[: size - len(chosen)]:
            pick = rng.choice(remaining[parent])
            remaining[parent].remove(pick)
            chosen.append(
                {
                    "sample_id": pick,
                    "parent_id": parent,
                    "image_id": image_of[pick],
                    "round": round_index,
                    "parent_population_size": len(by_parent[parent]),
                }
            )
    n_parents = len(by_parent)
    for row in chosen:
        if round_index == 1:
            row["inclusion_probability"] = (size / n_parents) / row["parent_population_size"]
        else:
            row["inclusion_probability"] = ""
    return chosen


def clustering_summary(sample: list[str], image_of: dict[str, str]) -> dict[str, Any]:
    images = Counter(image_of[s] for s in sample)
    parents = Counter(parent_of(image_of[s])[0] for s in sample)

    def describe(counter: Counter) -> dict[str, Any]:
        sizes = Counter(counter.values())
        return {
            "distinct": len(counter),
            "max_per_cluster": max(counter.values()),
            "cluster_size_histogram": {str(k): sizes[k] for k in sorted(sizes)},
            "samples_in_clusters_ge2": sum(v for v in counter.values() if v >= 2),
            "top_clusters": [[name, count] for name, count in counter.most_common(5) if count >= 2],
        }

    return {"n": len(sample), "by_image": describe(images), "by_parent": describe(parents)}


def expected_random_clustering(
    population: list[str],
    image_of: dict[str, str],
    size: int,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Reference distribution of distinct images/parents under simple random sampling."""
    rng = random.Random(seed)
    pool = sorted(population)
    distinct_images, distinct_parents, max_parent = [], [], []
    for _ in range(draws):
        sample = rng.sample(pool, size)
        distinct_images.append(len({image_of[s] for s in sample}))
        parents = Counter(parent_of(image_of[s])[0] for s in sample)
        distinct_parents.append(len(parents))
        max_parent.append(max(parents.values()))
    return {
        "draws": draws,
        "distinct_images_mean": round(float(np.mean(distinct_images)), 2),
        "distinct_images_p05_p95": [int(np.percentile(distinct_images, 5)), int(np.percentile(distinct_images, 95))],
        "distinct_parents_mean": round(float(np.mean(distinct_parents)), 2),
        "distinct_parents_p05_p95": [int(np.percentile(distinct_parents, 5)), int(np.percentile(distinct_parents, 95))],
        "max_per_parent_p05_p95": [int(np.percentile(max_parent, 5)), int(np.percentile(max_parent, 95))],
    }


def crop_window(bbox, width: int, height: int, factor: float, min_side: float) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    side = max(min_side, max(w, h) * factor)
    crop_w, crop_h = min(float(width), side), min(float(height), side)
    cx, cy = x + w / 2.0, y + h / 2.0
    x1 = min(max(0.0, cx - crop_w / 2.0), width - crop_w)
    y1 = min(max(0.0, cy - crop_h / 2.0), height - crop_h)
    x1_i, y1_i = int(np.floor(max(0.0, x1))), int(np.floor(max(0.0, y1)))
    return x1_i, y1_i, min(width, int(np.ceil(x1 + crop_w))), min(height, int(np.ceil(y1 + crop_h)))


def raw_crop_tile(image: np.ndarray, bbox, factor: float, min_side: float) -> np.ndarray:
    """Raw pixels; target extent marked only by ticks in the margin, never on the pixels."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = crop_window(bbox, width, height, factor, min_side)
    crop = image[y1:y2, x1:x2]
    inner = TILE_PX - 2 * TILE_MARGIN_PX
    scale = min(inner / crop.shape[1], inner / crop.shape[0])
    new_w, new_h = max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    tile = np.full((TILE_PX, TILE_PX, 3), COLOR_BG, dtype=np.uint8)
    pad_x, pad_y = (TILE_PX - new_w) // 2, (TILE_PX - new_h) // 2
    tile[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    bx, by, bw, bh = bbox
    tx1 = int(round(pad_x + (max(bx, x1) - x1) * scale))
    tx2 = int(round(pad_x + (min(bx + bw, x2) - x1) * scale))
    ty1 = int(round(pad_y + (max(by, y1) - y1) * scale))
    ty2 = int(round(pad_y + (min(by + bh, y2) - y1) * scale))
    tick = TILE_MARGIN_PX - 4
    top, bottom = pad_y - 2, pad_y + new_h + 1
    left, right = pad_x - 2, pad_x + new_w + 1
    for tx in (tx1, tx2):
        cv2.line(tile, (tx, max(0, top - tick)), (tx, top), COLOR_TARGET, 2)
        cv2.line(tile, (tx, bottom), (tx, min(TILE_PX - 1, bottom + tick)), COLOR_TARGET, 2)
    for ty in (ty1, ty2):
        cv2.line(tile, (max(0, left - tick), ty), (left, ty), COLOR_TARGET, 2)
        cv2.line(tile, (right, ty), (min(TILE_PX - 1, right + tick), ty), COLOR_TARGET, 2)
    return tile


def render_blind_panel(image: np.ndarray, bbox, blind_id: str) -> np.ndarray:
    full = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(full, (int(round(x)), int(round(y))), (int(round(x + w)), int(round(y + h))), COLOR_TARGET, 2, cv2.LINE_AA)
    if full.shape[:2] != (PANEL_SIDE_PX, PANEL_SIDE_PX):
        full = cv2.resize(full, (PANEL_SIDE_PX, PANEL_SIDE_PX), interpolation=cv2.INTER_AREA)
    right = np.vstack([raw_crop_tile(image, bbox, *CROP_NEAR), raw_crop_tile(image, bbox, *CROP_WIDE)])
    body = np.hstack([full, right])
    header = np.full((HEADER_PX, body.shape[1], 3), COLOR_BG, dtype=np.uint8)
    cv2.putText(header, blind_id, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(
        header,
        "left: full patch, target box   right: raw context ~2x (top) / ~4x (bottom), ticks mark target extent",
        (170, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLOR_TEXT_DIM,
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, body])


def build_detection_records(
    sample_ids: set[str],
    index_df: pd.DataFrame,
    predictions_by_image: dict[str, list[dict[str, Any]]],
    greedy_matches: dict[str, Any],
    gt_cache: dict[str, list[tuple[float, float, float, float]]],
    annotations_root: Path,
    images_root: Path,
    png_index: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Per-sample geometry and diagnostics (diagnostics go only to the sealed lookup)."""
    records: dict[str, dict[str, Any]] = {}
    for image_name, group in index_df.groupby("image_name", sort=True):
        image_name = str(image_name)
        group_ids = [str(s) for s in group["sample_id"]]
        if not sample_ids.intersection(group_ids):
            continue
        image_path = resolve_patch_image_from_index(images_root, image_name, png_index)
        if image_path is None or cv2.imread(str(image_path)) is None:
            fail(f"could not read image for {image_name}")
        height, width = cv2.imread(str(image_path)).shape[:2]
        gt_bboxes = gt_cache.get(image_name, [])
        centers = _audit.load_center_points(resolve_labelme_json(annotations_root, image_name))
        detections = {}
        for _, sample in group.iterrows():
            bbox, _ = _audit.resolve_yolo_bbox_and_confidence(sample, predictions_by_image)
            detections[str(sample["sample_id"])] = {"bbox": bbox, "confidence": float(sample["confidence"])}

        for sample_id in group_ids:
            if sample_id not in sample_ids:
                continue
            bbox = detections[sample_id]["bbox"]
            match = greedy_matches[sample_id]
            near_index, near_iou, _ = best_gt_overlap(bbox, gt_bboxes)
            other_iou, other_id = 0.0, ""
            for other_sid, other in detections.items():
                if other_sid == sample_id:
                    continue
                overlap = iou_xywh(bbox, other["bbox"])
                if overlap > other_iou:
                    other_iou, other_id = overlap, other_sid
            parent, position = parent_of(image_name)
            records[sample_id] = {
                "sample_id": sample_id,
                "image_id": image_name,
                "image_path": image_path,
                "parent_id": parent,
                "patch_position": position,
                "bbox": bbox,
                "yolo_confidence": round(detections[sample_id]["confidence"], 6),
                "bbox_x": round(bbox[0], 3),
                "bbox_y": round(bbox[1], 3),
                "bbox_w": round(bbox[2], 3),
                "bbox_h": round(bbox[3], 3),
                "relative_bbox_area": round(bbox[2] * bbox[3] / float(width * height), 6),
                "protocol_v2_matched_gt": "true" if match.matched_gt else "false",
                "protocol_v2_gt_index": "" if match.gt_index is None else int(match.gt_index),
                "protocol_v2_iou": round(float(match.max_iou), 4),
                "nearest_gt_iou": round(float(near_iou), 4),
                "nearest_gt_index": "" if near_index is None else int(near_index),
                "num_gt_palms": len(gt_bboxes),
                "num_yolo_detections_ge_05": len(detections),
                "touches_border": "true" if _audit.touches_border(bbox, width, height) else "false",
                "center_points_inside_target": sum(_audit.point_in_bbox(p, bbox) for p in centers),
                "nearest_other_detection_iou": round(other_iou, 4) if len(detections) > 1 else "",
                "nearest_other_detection_sample_id": other_id,
                "nearest_other_detection_matched_gt": ""
                if not other_id
                else ("true" if greedy_matches[other_id].matched_gt else "false"),
            }
    return records


def export_sample(
    *,
    output_dir: Path,
    id_prefix: str,
    population_name: str,
    draw: list[str],
    review_order: list[str],
    records: dict[str, dict[str, Any]],
    audit_index_of: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    draw_position = {sample_id: position for position, sample_id in enumerate(draw, start=1)}
    images_in_sample = Counter(records[s]["image_id"] for s in draw)
    parents_in_sample = Counter(records[s]["parent_id"] for s in draw)

    manifest_rows, lookup_rows = [], []
    for position, sample_id in enumerate(review_order, start=1):
        blind_id = f"{id_prefix}_{position:03d}"
        record = records[sample_id]
        image = cv2.imread(str(record["image_path"]))
        panel_path = panels_dir / f"{blind_id}.png"
        if not cv2.imwrite(str(panel_path), render_blind_panel(image, record["bbox"], blind_id),
                           [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
            fail(f"failed to write {panel_path}")
        manifest_rows.append(
            {
                "blind_id": blind_id,
                "sample_id": sample_id,
                "blind_visualization_path": str(panel_path.resolve().relative_to(PROJECT_ROOT)),
                "semantic": "",
                "reviewer_confidence": "",
                "notes": "",
            }
        )
        lookup_rows.append(
            {
                **{column: record.get(column, "") for column in LOOKUP_COLUMNS},
                "blind_id": blind_id,
                "review_position": position,
                "draw_position": draw_position[sample_id],
                "population": population_name,
                "audit_index": audit_index_of.get(sample_id, ""),
                "same_image_in_sample": images_in_sample[record["image_id"]],
                "same_parent_in_sample": parents_in_sample[record["parent_id"]],
            }
        )
    manifest = pd.DataFrame(manifest_rows, columns=list(BLIND_MANIFEST_COLUMNS))
    lookup = pd.DataFrame(lookup_rows, columns=list(LOOKUP_COLUMNS))
    manifest.to_csv(output_dir / "blind_manifest.csv", index=False)
    lookup.to_csv(output_dir / "diagnostic_lookup.csv", index=False)
    return manifest, lookup


REVIEWER_README = """# Blind semantic review: {title}

Review set of {n} YOLO detections, each shown as one panel in `panels/`.
You only need `blind_manifest.csv` and `panels/`.

## Do not open until all {n} rows are labelled

- `diagnostic_lookup.csv` (sealed)
- `sampling_report.json`, `SAMPLING_METHODS.md`{extra_sealed}
- anything under `outputs/evaluation*`, `outputs/verification*`, `outputs/diagnostics/gt_negative_audit/`
  or other diagnostics directories, and any VLM output

## Question

For each panel answer one question only:

> Does the target (yellow box on the left; ticks on the right-hand crops) visually
> correspond to a genuine palm?

## Panel

- Left: full patch with only the target box.
- Right top: raw pixels, ~2x context around the target.
- Right bottom: raw pixels, ~4x context around the target.
- Yellow ticks in the crop margins mark the target's horizontal and vertical extent;
  crops are centred on the target unless the patch edge forces a shift.

## Fill in `blind_manifest.csv` (edit only these three columns)

| Column | Allowed values |
|---|---|
| `semantic` | `palm`, `non_palm`, `ambiguous` |
| `reviewer_confidence` | `high`, `low` |
| `notes` | free text (optional) |

Guidance:

- `palm`: the box is on a real palm crown, including partially visible or edge-cut palms
  and boxes that fit the palm loosely.
- `non_palm`: the box is on something that is not a palm (other tree, shrub, shadow,
  ground, structure).
- `ambiguous`: you cannot decide from the image.
- Judge the object in the box, not the tightness of the box; describe box problems in `notes`.
- Review in `blind_id` order, keep every row, and do not edit `blind_id`, `sample_id`
  or `blind_visualization_path`. Do not look up `sample_id` anywhere.
"""


def write_methods(
    output_dir: Path,
    *,
    population_name: str,
    population_size: int,
    sample_seed: int,
    order_seed: int,
    id_prefix: str,
    report: dict[str, Any],
    cluster_note: str,
) -> None:
    lines = [
        f"# Sampling methods: {population_name} (SEALED until review is complete)",
        "",
        f"Evaluation Protocol {EVALUATION_PROTOCOL_VERSION}; IoU threshold {IOU_THRESHOLD}; population recomputed with",
        "the official greedy one-to-one matching and cross-checked against every stored full-size",
        "Protocol v2 evaluation CSV. No VLM prediction was read; no inference was run.",
        "",
        "## Population",
        "",
        f"All {population_size} current {population_name} verification detections, as a list of",
        "`sample_id` strings sorted ascending. No other attribute enters the draw.",
        "",
        "## Primary sample (simple random, without replacement)",
        "",
        "```python",
        f"draw = random.Random({sample_seed}).sample(sorted(population_sample_ids), {SAMPLE_SIZE})",
        "```",
        "",
        "`draw_position` in `diagnostic_lookup.csv` is the position in `draw`.",
        "",
        "## Review order and blind IDs",
        "",
        "```python",
        "order = sorted(draw)",
        f"random.Random({order_seed}).shuffle(order)",
        f"blind_id = f\"{id_prefix}_{{position:03d}}\"   # position = 1..{SAMPLE_SIZE} in order",
        "```",
        "",
        "## Parent images",
        "",
        "Patch names are `<flight>_<folder>_<frame>_<k>` with `k` in 1..8; the eight patches of a",
        "frame form a 2x4 grid (k 1-4 top row, 5-8 bottom row). Verified on all 48 frames with",
        "all eight patches present: every expected neighbour edge (mean abs pixel difference,",
        "median 13.3, max 23.2) is below the median of non-neighbour edges (46.0). `parent_id`",
        "= name without the final `_<k>`.",
        "",
        "## Clustering (see `sampling_report.json`)",
        "",
        cluster_note,
        "",
    ]
    (output_dir / "SAMPLING_METHODS.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "sampling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def cluster_note_for(summary: dict[str, Any], reference: dict[str, Any]) -> str:
    by_image, by_parent = summary["by_image"], summary["by_parent"]
    return (
        f"Seed sample: {by_image['distinct']} distinct images (max {by_image['max_per_cluster']} per image), "
        f"{by_parent['distinct']} distinct parents (max {by_parent['max_per_cluster']} per parent). "
        f"Reference over {reference['draws']} other simple random draws: distinct images mean "
        f"{reference['distinct_images_mean']} (5-95% {reference['distinct_images_p05_p95']}), distinct parents mean "
        f"{reference['distinct_parents_mean']} (5-95% {reference['distinct_parents_p05_p95']}), max per parent "
        f"5-95% {reference['max_per_parent_p05_p95']}. The primary sample was not altered."
    )


def validate(
    *,
    output_dir: Path,
    manifest: pd.DataFrame,
    lookup: pd.DataFrame,
    population: list[str],
    sample_seed: int,
    order_seed: int,
    id_prefix: str,
    matched_by_sample: dict[str, bool],
    expected_matched: bool,
) -> dict[str, bool]:
    reloaded = pd.read_csv(output_dir / "blind_manifest.csv", dtype=str, keep_default_na=False)
    redraw = simple_random_sample(population, SAMPLE_SIZE, sample_seed)
    reorder = shuffled_review_order(redraw, order_seed)
    lowered = [column.lower() for column in reloaded.columns]
    panels = sorted(p.name for p in (output_dir / "panels").glob("*.png"))
    checks = {
        f"exactly {SAMPLE_SIZE} rows": len(reloaded) == SAMPLE_SIZE,
        "sample_id unique": reloaded["sample_id"].is_unique,
        f"all samples in the {len(population)}-detection population": set(reloaded["sample_id"]) <= set(population),
        f"all samples have matched_gt={expected_matched}": all(
            matched_by_sample[s] == expected_matched for s in reloaded["sample_id"]
        ),
        f"seed {sample_seed} draw reproducible": sorted(redraw) == sorted(reloaded["sample_id"]),
        f"review order reproducible (seed {order_seed})": list(reloaded["sample_id"]) == reorder,
        "review order differs from sample_id order": list(reloaded["sample_id"]) != sorted(reloaded["sample_id"]),
        "blind IDs sequential": list(reloaded["blind_id"]) == [f"{id_prefix}_{i:03d}" for i in range(1, SAMPLE_SIZE + 1)],
        f"exactly {SAMPLE_SIZE} panel files": len(panels) == SAMPLE_SIZE,
        "panel filenames are blind IDs only": panels == [f"{b}.png" for b in reloaded["blind_id"]],
        "every blind_visualization_path exists": all(
            (PROJECT_ROOT / p).is_file() for p in reloaded["blind_visualization_path"]
        ),
        "manifest columns are exactly the allowed set": tuple(reloaded.columns) == BLIND_MANIFEST_COLUMNS,
        "no diagnostic/VLM column in manifest": not any(
            token in column for column in lowered if column not in {"blind_id", "sample_id", "blind_visualization_path",
                                                                    "semantic", "reviewer_confidence", "notes"}
            for token in FORBIDDEN_MANIFEST_TOKENS
        ),
        "review columns all empty": all((reloaded[c] == "").all() for c in REVIEW_COLUMNS),
        "lookup has same blind_id/sample_id pairs": list(lookup["blind_id"]) == list(reloaded["blind_id"])
        and list(lookup["sample_id"]) == list(reloaded["sample_id"]),
        "lookup has no semantic/VLM columns": not any(
            token in column.lower() for column in lookup.columns
            for token in ("semantic", "reviewer", "vlm", "decision", "reliable", "verification_label", "priority")
        ),
    }
    return checks


def main() -> None:
    args = parse_args()
    for path in (args.index_csv, args.predictions, args.annotations_root, args.images_root, EXISTING_AUDIT_MANIFEST):
        if not path.exists():
            fail(f"required input not found: {path}")

    index_df = pd.read_csv(args.index_csv)
    if index_df["sample_id"].duplicated().any():
        fail("duplicate sample_id values in the verification index")
    predictions_by_image = group_predictions_by_image(load_predictions(args.predictions))
    gt_cache: dict[str, list[tuple[float, float, float, float]]] = {}
    greedy_matches = _eval.compute_greedy_matches_for_index(
        index_df, predictions_by_image, args.annotations_root, gt_cache, IOU_THRESHOLD
    )
    matched_by_sample = {s: m.matched_gt for s, m in greedy_matches.items()}
    negatives = sorted(s for s, m in matched_by_sample.items() if not m)
    positives = sorted(s for s, m in matched_by_sample.items() if m)
    counts = (len(index_df), len(positives), len(negatives))
    print(f"Verification detections: {counts[0]}  GT+: {counts[1]}  GT-: {counts[2]}")
    if counts != (EXPECTED_TOTAL, EXPECTED_MATCHED, EXPECTED_UNMATCHED):
        fail(f"expected {EXPECTED_TOTAL}/{EXPECTED_MATCHED}/{EXPECTED_UNMATCHED}, got {counts}")
    n_checked = _audit.cross_check_stored_evaluations(args.evaluation_root, matched_by_sample)
    if n_checked == 0:
        fail("no stored full-size evaluation CSV found for the cross-check")
    print(f"matched_gt identical to {n_checked} stored Protocol v2 evaluation CSVs")

    audit_manifest = pd.read_csv(EXISTING_AUDIT_MANIFEST, dtype=str, keep_default_na=False)
    if sorted(audit_manifest["sample_id"]) != negatives:
        fail("existing gt_negative_audit manifest does not list exactly the current GT- population")
    audit_index_of = {row.sample_id: int(row.audit_index) for row in audit_manifest.itertuples()}

    image_of = {str(r.sample_id): str(r.image_name) for r in index_df.itertuples()}

    negative_draw = simple_random_sample(negatives, SAMPLE_SIZE, NEGATIVE_SAMPLE_SEED)
    negative_order = shuffled_review_order(negative_draw, NEGATIVE_REVIEW_ORDER_SEED)
    positive_draw = simple_random_sample(positives, SAMPLE_SIZE, POSITIVE_SAMPLE_SEED)
    positive_order = shuffled_review_order(positive_draw, POSITIVE_REVIEW_ORDER_SEED)
    cluster_rows = cluster_aware_sample(negatives, image_of, SAMPLE_SIZE, CLUSTER_AWARE_SEED)

    prepare_output_dir(NEGATIVE_DIR, args.overwrite)
    prepare_output_dir(POSITIVE_DIR, args.overwrite)

    png_index = build_png_index(args.images_root)
    records = build_detection_records(
        set(negative_draw) | set(positive_draw),
        index_df,
        predictions_by_image,
        greedy_matches,
        gt_cache,
        args.annotations_root,
        args.images_root,
        png_index,
    )

    results = {}
    for (output_dir, prefix, name, population, draw, order, sample_seed, order_seed, expected, title) in (
        (NEGATIVE_DIR, "blind", "GT-negative (matched_gt=False)", negatives, negative_draw, negative_order,
         NEGATIVE_SAMPLE_SEED, NEGATIVE_REVIEW_ORDER_SEED, False, "pilot set"),
        (POSITIVE_DIR, "qc", "GT-positive (matched_gt=True)", positives, positive_draw, positive_order,
         POSITIVE_SAMPLE_SEED, POSITIVE_REVIEW_ORDER_SEED, True, "QC set"),
    ):
        manifest, lookup = export_sample(
            output_dir=output_dir,
            id_prefix=prefix,
            population_name=name,
            draw=draw,
            review_order=order,
            records=records,
            audit_index_of=audit_index_of,
        )
        summary = clustering_summary(draw, image_of)
        population_summary = {
            "n": len(population),
            "distinct_images": len({image_of[s] for s in population}),
            "distinct_parents": len({parent_of(image_of[s])[0] for s in population}),
        }
        reference = expected_random_clustering(population, image_of, SAMPLE_SIZE)
        report = {
            "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
            "population": name,
            "population_summary": population_summary,
            "sample_seed": sample_seed,
            "review_order_seed": order_seed,
            "sample_clustering": summary,
            "simple_random_reference": reference,
            "stored_evaluation_csvs_cross_checked": n_checked,
        }
        is_negative = output_dir == NEGATIVE_DIR
        if is_negative:
            cluster_df = pd.DataFrame(cluster_rows)
            cluster_df.insert(0, "cluster_aware_position", range(1, len(cluster_df) + 1))
            cluster_df["in_primary_sample"] = cluster_df["sample_id"].isin(set(draw)).map({True: "true", False: "false"})
            cluster_df.to_csv(output_dir / "cluster_aware_manifest.csv", index=False)
            report["cluster_aware_secondary"] = {
                "seed": CLUSTER_AWARE_SEED,
                "rounds_used": int(cluster_df["round"].max()),
                "clustering": clustering_summary(list(cluster_df["sample_id"]), image_of),
                "overlap_with_primary": int((cluster_df["in_primary_sample"] == "true").sum()),
            }
        write_methods(
            output_dir,
            population_name=name,
            population_size=len(population),
            sample_seed=sample_seed,
            order_seed=order_seed,
            id_prefix=prefix,
            report=report,
            cluster_note=cluster_note_for(summary, reference),
        )
        (output_dir / "README.md").write_text(
            REVIEWER_README.format(
                title=title,
                n=SAMPLE_SIZE,
                extra_sealed=", `cluster_aware_manifest.csv`" if is_negative else "",
            ),
            encoding="utf-8",
        )
        checks = validate(
            output_dir=output_dir,
            manifest=manifest,
            lookup=lookup,
            population=population,
            sample_seed=sample_seed,
            order_seed=order_seed,
            id_prefix=prefix,
            matched_by_sample=matched_by_sample,
            expected_matched=expected,
        )
        results[output_dir.name] = (checks, report)

    checks = {
        "negative and positive samples disjoint": not set(negative_draw) & set(positive_draw),
        "existing audit manifest still has 638 rows": len(pd.read_csv(EXISTING_AUDIT_MANIFEST)) == EXPECTED_UNMATCHED,
    }
    results["cross"] = (checks, None)

    all_passed = True
    for name, (checks, report) in results.items():
        print(f"\nValidation: {name}")
        for check, passed in checks.items():
            all_passed &= bool(passed)
            print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
        if report:
            print(json.dumps(report["sample_clustering"], indent=2))
            print("reference:", json.dumps(report["simple_random_reference"]))
            if "cluster_aware_secondary" in report:
                print("cluster-aware:", json.dumps(report["cluster_aware_secondary"], indent=2))
    if not all_passed:
        print("\nValidation FAILED; do not use these directories.")
        sys.exit(1)
    print(f"\nWrote {NEGATIVE_DIR} and {POSITIVE_DIR}")


if __name__ == "__main__":
    main()
