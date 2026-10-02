"""
SUPERSEDED / NOT FOR CURRENT RESULTS: this module predates the official semantic review
workflow, reads the never-filled outputs/semantic_gt_evaluation/review manifest instead of
outputs/semantic_gt_review/human_review.csv, and expects fields that were never collected.
Canonical semantic results: docs/SEMANTIC_VALIDITY_AUDIT.md.

Semantic ground truth (human-reviewed object presence) for YOLO verification detections.

This is a second GT definition that lives alongside Evaluation Protocol v2; it never
replaces or edits Protocol v2 labels. Protocol v2 GT answers "is this detection matched
to a LabelMe palm box at IoU >= 0.5?"; semantic GT answers "does this detection
visually correspond to a palm?". See docs/SEMANTIC_GT_EVALUATION.md.

Label sources (``label_provenance``):

- ``inherited_positive``: Protocol v2 GT+ detections, assigned ``semantic_gt = palm``
  without manual review.
- ``manually_reviewed_negative_pool``: Protocol v2 GT- detections with a human label
  in the semantic review manifest.
- ``unreviewed_negative_pool``: Protocol v2 GT- detections whose manifest row is still
  blank. These carry no semantic label and are never counted as negatives.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paths import OUTPUTS_DIR

SEMANTIC_GT_ROOT = OUTPUTS_DIR / "semantic_gt_evaluation"
SEMANTIC_REVIEW_MANIFEST_CSV = SEMANTIC_GT_ROOT / "review" / "semantic_review_manifest.csv"
PROTOCOL_V2_REFERENCE_CSV = SEMANTIC_GT_ROOT / "reference" / "protocol_v2_detection_reference.csv"

DETECTION_KEY = "sample_id"

EXPECTED_TOTAL = 5747
EXPECTED_V2_POSITIVE = 5109
EXPECTED_V2_NEGATIVE = 638

PALM = "palm"
NON_PALM = "non_palm"
AMBIGUOUS = "ambiguous"
UNREVIEWED = ""
SEMANTIC_LABELS = (PALM, NON_PALM, AMBIGUOUS)
BINARY_SEMANTIC_LABELS = (PALM, NON_PALM)

MATCHED_EXISTING_GT = "matched_existing_gt"
MISSING_ANNOTATION = "missing_annotation"
LOCALIZATION_MISMATCH = "localization_mismatch"
DUPLICATE_DETECTION = "duplicate_detection"
TRUE_NON_PALM = "true_non_palm"
REASON_AMBIGUOUS = "ambiguous"
OTHER = "other"
REVIEW_REASONS = (
    MATCHED_EXISTING_GT,
    MISSING_ANNOTATION,
    LOCALIZATION_MISMATCH,
    DUPLICATE_DETECTION,
    TRUE_NON_PALM,
    REASON_AMBIGUOUS,
    OTHER,
)
# matched_existing_gt is reserved for inherited positives and is never a manual reason.
ALLOWED_REASONS_BY_LABEL = {
    PALM: {MISSING_ANNOTATION, LOCALIZATION_MISMATCH, DUPLICATE_DETECTION, OTHER},
    NON_PALM: {TRUE_NON_PALM, OTHER},
    AMBIGUOUS: {REASON_AMBIGUOUS, OTHER},
}

INHERITED_POSITIVE = "inherited_positive"
MANUALLY_REVIEWED_NEGATIVE_POOL = "manually_reviewed_negative_pool"
UNREVIEWED_NEGATIVE_POOL = "unreviewed_negative_pool"

V2_POSITIVE = "positive"
V2_NEGATIVE = "negative"

REFERENCE_COLUMNS = [
    "sample_id",
    "image_id",
    "image_path",
    "yolo_bbox_xywh",
    "yolo_confidence",
    "max_iou",
    "matched_gt_index",
    "nearest_gt_index",
    "nearest_gt_bbox_xywh",
    "nearest_gt_owner_sample_id",
    "original_protocol_v2_gt",
]

MANUAL_COLUMNS = ["semantic_gt", "review_reason", "reviewer_notes"]
REVIEW_MANIFEST_COLUMNS = [
    "review_index",
    "sample_id",
    "image_id",
    "image_path",
    "yolo_bbox_xywh",
    "yolo_confidence",
    "max_iou",
    "nearest_gt_index",
    "nearest_gt_bbox_xywh",
    "nearest_gt_owner_sample_id",
    "iou_ge_050_gt_taken_by_other",
    "original_protocol_v2_gt",
    "audit_panel_path",
    *MANUAL_COLUMNS,
]
# Computed columns that must match the reference exactly; reviewers edit MANUAL_COLUMNS only.
MANIFEST_IDENTITY_COLUMNS = ["image_id", "yolo_bbox_xywh", "original_protocol_v2_gt"]

SEMANTIC_TABLE_COLUMNS = [*REFERENCE_COLUMNS, *MANUAL_COLUMNS, "label_provenance"]


class SemanticGtError(ValueError):
    """Raised when semantic GT inputs violate the schema or join invariants."""


def read_str_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_reference(path: Path = PROTOCOL_V2_REFERENCE_CSV) -> pd.DataFrame:
    reference = read_str_csv(path)
    validate_reference(reference)
    return reference


def validate_reference(
    reference: pd.DataFrame,
    expected_counts: tuple[int, int] | None = (EXPECTED_V2_POSITIVE, EXPECTED_V2_NEGATIVE),
) -> None:
    missing = [c for c in REFERENCE_COLUMNS if c not in reference.columns]
    if missing:
        raise SemanticGtError(f"reference missing columns: {missing}")
    if not reference[DETECTION_KEY].is_unique:
        raise SemanticGtError("reference has duplicate sample_id values")
    values = set(reference["original_protocol_v2_gt"])
    if not values <= {V2_POSITIVE, V2_NEGATIVE}:
        raise SemanticGtError(f"reference original_protocol_v2_gt values {values}")
    if expected_counts is not None:
        positives = int((reference["original_protocol_v2_gt"] == V2_POSITIVE).sum())
        counts = (positives, len(reference) - positives)
        if len(reference) != sum(expected_counts) or counts != expected_counts:
            raise SemanticGtError(
                f"reference has {len(reference)} rows, GT+/GT- {counts}; "
                f"expected {sum(expected_counts)} rows, {expected_counts}"
            )


def _normalize(value: object) -> str:
    return str(value if value is not None else "").strip()


def normalize_manual_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace; lowercase the two controlled-vocabulary columns."""
    out = manifest.copy()
    for column in MANUAL_COLUMNS:
        out[column] = out[column].map(_normalize)
    out["semantic_gt"] = out["semantic_gt"].str.lower()
    out["review_reason"] = out["review_reason"].str.lower()
    return out


def validate_review_manifest(manifest: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a (possibly partially filled) semantic review manifest against the
    Protocol v2 reference and return it with normalized manual columns.

    Raises SemanticGtError listing every violation found.
    """
    errors: list[str] = []
    missing = [c for c in REVIEW_MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise SemanticGtError(f"manifest missing columns: {missing}")

    manifest = normalize_manual_columns(manifest)
    keys = manifest[DETECTION_KEY].map(_normalize)

    duplicated = sorted(set(keys[keys.duplicated()]))
    if duplicated:
        errors.append(f"duplicate sample_id rows: {duplicated[:10]}")

    ref_by_key = reference.set_index(DETECTION_KEY)
    unknown = sorted(set(keys) - set(ref_by_key.index))
    if unknown:
        errors.append(f"sample_id not in canonical detection set: {unknown[:10]}")

    negatives = set(reference.loc[reference["original_protocol_v2_gt"] == V2_NEGATIVE, DETECTION_KEY])
    known = keys[keys.isin(ref_by_key.index)]
    not_negative = sorted(set(known) - negatives)
    if not_negative:
        errors.append(f"rows that are not Protocol v2 GT-: {not_negative[:10]}")
    absent = sorted(negatives - set(keys))
    if absent:
        errors.append(f"{len(absent)} Protocol v2 GT- detections missing from manifest: {absent[:10]}")

    if not unknown:
        for column in MANIFEST_IDENTITY_COLUMNS:
            expected = ref_by_key.loc[keys, column].to_numpy()
            actual = manifest[column].map(_normalize).to_numpy()
            bad = keys[expected != actual]
            if len(bad):
                errors.append(f"column {column} differs from reference for {list(bad[:10])}")

    bad_labels = sorted(set(manifest["semantic_gt"]) - {*SEMANTIC_LABELS, UNREVIEWED})
    if bad_labels:
        errors.append(f"semantic_gt values not allowed: {bad_labels} (allowed {SEMANTIC_LABELS} or blank)")
    bad_reasons = sorted(set(manifest["review_reason"]) - {*REVIEW_REASONS, ""})
    if bad_reasons:
        errors.append(f"review_reason values not allowed: {bad_reasons}")

    for key, label, reason in zip(keys, manifest["semantic_gt"], manifest["review_reason"]):
        if label not in (*SEMANTIC_LABELS, UNREVIEWED) or (reason and reason not in REVIEW_REASONS):
            continue
        if label == UNREVIEWED and reason:
            errors.append(f"{key}: review_reason {reason!r} without semantic_gt")
        elif label and not reason:
            errors.append(f"{key}: semantic_gt {label!r} without review_reason")
        elif label and reason not in ALLOWED_REASONS_BY_LABEL[label]:
            errors.append(f"{key}: review_reason {reason!r} inconsistent with semantic_gt {label!r}")

    if errors:
        raise SemanticGtError("invalid semantic review manifest:\n  " + "\n  ".join(errors[:50]))
    manifest[DETECTION_KEY] = keys
    return manifest


def build_semantic_gt_table(reference: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Protocol v2 reference and validated review manifest into one semantic GT
    row per canonical detection (same order as the reference).
    """
    manifest = validate_review_manifest(manifest, reference)
    table = reference[REFERENCE_COLUMNS].copy()
    manual = manifest.set_index(DETECTION_KEY)[MANUAL_COLUMNS]

    positive = table["original_protocol_v2_gt"] == V2_POSITIVE
    table["semantic_gt"] = ""
    table["review_reason"] = ""
    table["reviewer_notes"] = ""
    table.loc[positive, "semantic_gt"] = PALM
    table.loc[positive, "review_reason"] = MATCHED_EXISTING_GT

    negative_keys = table.loc[~positive, DETECTION_KEY]
    for column in MANUAL_COLUMNS:
        table.loc[~positive, column] = manual.loc[negative_keys, column].to_numpy()

    table["label_provenance"] = INHERITED_POSITIVE
    reviewed = ~positive & (table["semantic_gt"] != UNREVIEWED)
    table.loc[reviewed, "label_provenance"] = MANUALLY_REVIEWED_NEGATIVE_POOL
    table.loc[~positive & ~reviewed, "label_provenance"] = UNREVIEWED_NEGATIVE_POOL

    if len(table) != len(reference) or not table[DETECTION_KEY].is_unique:
        raise SemanticGtError("semantic table join did not produce one row per detection")
    return table[SEMANTIC_TABLE_COLUMNS]


def semantic_status(table: pd.DataFrame) -> dict[str, int | bool | str]:
    labels = table["semantic_gt"]
    provenance = table["label_provenance"]
    unreviewed = int((provenance == UNREVIEWED_NEGATIVE_POOL).sum())
    return {
        "n_total": len(table),
        "n_semantic_palm": int((labels == PALM).sum()),
        "n_semantic_palm_inherited": int((provenance == INHERITED_POSITIVE).sum()),
        "n_semantic_palm_reviewed": int(((labels == PALM) & (provenance == MANUALLY_REVIEWED_NEGATIVE_POOL)).sum()),
        "n_semantic_non_palm": int((labels == NON_PALM).sum()),
        "n_ambiguous": int((labels == AMBIGUOUS).sum()),
        "n_unreviewed": unreviewed,
        "n_negative_pool": int((table["original_protocol_v2_gt"] == V2_NEGATIVE).sum()),
        "n_negative_pool_reviewed": int((provenance == MANUALLY_REVIEWED_NEGATIVE_POOL).sum()),
        "semantic_gt_complete": unreviewed == 0,
        "semantic_gt_status": "complete" if unreviewed == 0 else "semantic GT incomplete",
    }


def semantic_binary_mask(table: pd.DataFrame) -> pd.Series:
    """Rows usable in semantic binary metrics: palm or non_palm (never ambiguous/unreviewed)."""
    return table["semantic_gt"].isin(BINARY_SEMANTIC_LABELS)


def negative_pool_category(label: str, reason: str) -> str:
    if label == UNREVIEWED:
        return "unreviewed"
    if label == PALM:
        return f"palm:{reason}"
    return label
