#!/usr/bin/env python3
"""
SUPERSEDED / NOT FOR CURRENT RESULTS: part of the earlier semantic-GT pipeline, which predates
the official review workflow (outputs/semantic_gt_review/human_review.csv). The manifest it
writes was never filled; see docs/SEMANTIC_VALIDITY_AUDIT.md.

Purpose:
    Build the Protocol v2 detection reference (all 5,747 canonical detections) and the
    blank semantic review manifest (the 638 Protocol v2 GT-negative detections) for
    the semantic-GT evaluation path. No inference; no existing file is modified.

    semantic_gt / review_reason / reviewer_notes are written blank. They are never
    inferred from filenames, YOLO confidence, IoU, VLM outputs or any model prediction.

Input:
    - outputs/verification_dataset/index.csv
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON under Raw_Patches
    - outputs/evaluation_protocol_v2/**/A*_evaluation.csv   (read-only cross-check)
    - outputs/diagnostics/gt_negative_audit/manifest.csv    (read-only cross-check, panels)

Output:
    - outputs/semantic_gt_evaluation/reference/protocol_v2_detection_reference.csv
    - outputs/semantic_gt_evaluation/review/semantic_review_manifest.csv

Example:
    python scripts/build_semantic_review_manifest.py
    python scripts/build_semantic_review_manifest.py --check-only
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_verification_against_groundtruth as eval_mod  # noqa: E402
from src.evaluation.semantic_gt import (  # noqa: E402
    MANUAL_COLUMNS,
    PROTOCOL_V2_REFERENCE_CSV,
    REFERENCE_COLUMNS,
    REVIEW_MANIFEST_COLUMNS,
    SEMANTIC_REVIEW_MANIFEST_CSV,
    V2_NEGATIVE,
    V2_POSITIVE,
    read_str_csv,
    validate_reference,
    validate_review_manifest,
)
from src.paths import (  # noqa: E402
    EVALUATION_PROTOCOL_V2_ROOT,
    OUTPUTS_DIR,
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.utils.verification_index import find_yolo_prediction, index_bbox_from_row  # noqa: E402
from src.yolo.predictions_io import group_predictions_by_image, load_predictions  # noqa: E402

GT_NEGATIVE_AUDIT_MANIFEST = OUTPUTS_DIR / "diagnostics" / "gt_negative_audit" / "manifest.csv"
V2_CROSS_CHECK_COLUMNS = {
    "yolo_bbox": "yolo_bbox_xywh",
    "yolo_confidence": "yolo_confidence",
    "max_iou": "max_iou",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference-out", type=Path, default=PROTOCOL_V2_REFERENCE_CSV)
    parser.add_argument("--manifest-out", type=Path, default=SEMANTIC_REVIEW_MANIFEST_CSV)
    parser.add_argument("--check-only", action="store_true",
                        help="Recompute and cross-check; write nothing")
    return parser.parse_args()


def build_reference() -> pd.DataFrame:
    index_df = pd.read_csv(VERIFICATION_DATASET_INDEX_CSV)
    predictions = group_predictions_by_image(load_predictions(PREDICTIONS_FULL_JSON))
    matches = eval_mod.compute_greedy_matches_for_index(
        index_df, predictions, RAW_PATCHES_ROOT, {}, eval_mod.IOU_THRESHOLD,
    )

    owner: dict[tuple[str, int], str] = {
        (str(row.image_name), matches[str(row.sample_id)].gt_index): str(row.sample_id)
        for row in index_df.itertuples()
        if matches[str(row.sample_id)].matched_gt
    }

    rows = []
    for _, sample in index_df.iterrows():
        sample_id = str(sample["sample_id"])
        image_id = str(sample["image_name"])
        index_bbox = index_bbox_from_row(sample)
        yolo_bbox, yolo_confidence = find_yolo_prediction(image_id, index_bbox, predictions)
        if yolo_bbox is None:
            yolo_bbox = index_bbox
            yolo_confidence = float(sample["confidence"])
        match = matches[sample_id]
        nearest_owner = (
            owner.get((image_id, match.gt_index), "") if match.gt_index is not None else ""
        )
        rows.append({
            "sample_id": sample_id,
            "image_id": image_id,
            "image_path": str(RAW_PATCHES_ROOT / f"{image_id}.png"),
            "yolo_bbox_xywh": eval_mod.bbox_to_json(yolo_bbox),
            "yolo_confidence": yolo_confidence,
            "max_iou": round(match.max_iou, 4),
            "matched_gt_index": match.gt_index if match.matched_gt else "",
            "nearest_gt_index": "" if match.gt_index is None else match.gt_index,
            "nearest_gt_bbox_xywh": eval_mod.bbox_to_json(match.gt_bbox),
            "nearest_gt_owner_sample_id": nearest_owner,
            "original_protocol_v2_gt": V2_POSITIVE if match.matched_gt else V2_NEGATIVE,
        })
    reference = pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
    # Round-trip through CSV so checks compare on-disk string representations.
    return pd.read_csv(io.StringIO(reference.to_csv(index=False)), dtype=str, keep_default_na=False)


def cross_check_v2_evaluations(reference: pd.DataFrame) -> int:
    """Every stored Protocol v2 evaluation CSV must agree with the recomputed reference."""
    csvs = sorted(
        path for path in EVALUATION_PROTOCOL_V2_ROOT.glob("*/*/A*/A*_evaluation.csv")
        if path.name == f"{path.parent.name}_evaluation.csv"
    )
    if not csvs:
        sys.exit(f"ABORT: no Protocol v2 evaluation CSVs under {EVALUATION_PROTOCOL_V2_ROOT}")
    expected_matched = reference["original_protocol_v2_gt"].eq(V2_POSITIVE).map(str)
    failures = []
    for path in csvs:
        v2 = read_str_csv(path)
        tag = str(path.relative_to(PROJECT_ROOT))
        if list(v2["sample_id"]) != list(reference["sample_id"]):
            failures.append(f"{tag}: sample_id order differs")
            continue
        if not v2["matched_gt"].equals(expected_matched):
            failures.append(f"{tag}: matched_gt differs")
        if not v2["image_name"].equals(reference["image_id"]):
            failures.append(f"{tag}: image_name differs")
        for v2_column, ref_column in V2_CROSS_CHECK_COLUMNS.items():
            if not v2[v2_column].equals(reference[ref_column]):
                failures.append(f"{tag}: {v2_column} differs")
    if failures:
        sys.exit("ABORT: reference disagrees with stored Protocol v2 outputs:\n  " + "\n  ".join(failures))
    return len(csvs)


def build_manifest(reference: pd.DataFrame) -> pd.DataFrame:
    audit = read_str_csv(GT_NEGATIVE_AUDIT_MANIFEST).set_index("sample_id")
    negatives = reference[reference["original_protocol_v2_gt"] == V2_NEGATIVE].reset_index(drop=True)
    if set(audit.index) != set(negatives["sample_id"]) or len(audit) != len(negatives):
        sys.exit("ABORT: existing gt_negative_audit manifest covers a different detection set")

    manifest = negatives[[c for c in REVIEW_MANIFEST_COLUMNS if c in negatives.columns]].copy()
    manifest.insert(0, "review_index", range(1, len(manifest) + 1))
    iou = pd.to_numeric(negatives["max_iou"])
    manifest["iou_ge_050_gt_taken_by_other"] = (iou >= eval_mod.IOU_THRESHOLD).map(
        {True: "true", False: "false"}
    )
    panels = audit.loc[negatives["sample_id"], "visualization_path"].to_numpy()
    missing_panels = [p for p in panels if not (PROJECT_ROOT / p).is_file()]
    if missing_panels:
        sys.exit(f"ABORT: {len(missing_panels)} audit panels missing, e.g. {missing_panels[:3]}")
    manifest["audit_panel_path"] = panels
    for column in MANUAL_COLUMNS:
        manifest[column] = ""
    taken = manifest["iou_ge_050_gt_taken_by_other"] == "true"
    if (manifest.loc[taken, "nearest_gt_owner_sample_id"] == "").any():
        sys.exit("ABORT: IoU >= 0.5 unmatched detection whose nearest GT has no owner")
    return manifest[REVIEW_MANIFEST_COLUMNS]


def write_reference(reference: pd.DataFrame, path: Path) -> None:
    if path.exists():
        if read_str_csv(path).equals(reference):
            print(f"Reference unchanged: {path}")
            return
        sys.exit(f"ABORT: existing reference differs from recomputation; not overwriting {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(path, index=False)
    print(f"Wrote reference: {path}")


def write_manifest(manifest: pd.DataFrame, path: Path, reference: pd.DataFrame) -> None:
    if path.exists():
        existing = validate_review_manifest(read_str_csv(path), reference)
        labelled = int((existing[MANUAL_COLUMNS] != "").any(axis=1).sum())
        print(f"Manifest exists ({labelled} rows with manual entries); not overwriting {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print(f"Wrote blank review manifest ({len(manifest)} rows): {path}")


def main() -> None:
    args = parse_args()
    reference = build_reference()
    validate_reference(reference)
    n_csvs = cross_check_v2_evaluations(reference)
    manifest = build_manifest(reference)
    validate_review_manifest(manifest, reference)

    counts = reference["original_protocol_v2_gt"].value_counts().to_dict()
    print(json.dumps({
        "detections": len(reference),
        "protocol_v2_gt": counts,
        "v2_evaluation_csvs_cross_checked": n_csvs,
        "review_manifest_rows": len(manifest),
        "iou_ge_050_gt_taken_by_other": int((manifest["iou_ge_050_gt_taken_by_other"] == "true").sum()),
    }, indent=2))
    if args.check_only:
        return
    write_reference(reference, args.reference_out)
    write_manifest(manifest, args.manifest_out, reference)


if __name__ == "__main__":
    main()
