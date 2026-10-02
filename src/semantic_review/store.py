"""
Review items, startup validation and the human-label store for the blind semantic
review tool.

Three modes, never mixed:

- ``unmatched``: the 638 Protocol v2 GT- detections from the semantic review manifest
  (outputs/semantic_gt_evaluation/review/semantic_review_manifest.csv).
- ``positive-qc``: the existing 100-row matched-positive blind QC sample
  (outputs/diagnostics/gt_positive_blind_qc/blind_manifest.csv), boxes taken from the
  Protocol v2 detection reference.
- ``confidence-pilot``: the 400-row lower-confidence blind pilot
  (outputs/semantic_candidate_audit/semantic_pilot_manifest.csv). Only the blind manifest
  is read; the sealed semantic_pilot_reference.csv is refused by path.

Source manifests are read only. Labels go to a dedicated CSV under
outputs/semantic_gt_review/ that holds identity columns plus ``semantic_label``,
``reviewer`` and ``review_timestamp``; no diagnostic column (confidence, IoU, nearest GT,
Protocol v2 class) is copied into it.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.evaluation.semantic_gt import (
    AMBIGUOUS,
    EXPECTED_V2_NEGATIVE,
    EXPECTED_V2_POSITIVE,
    NON_PALM,
    PALM,
    PROTOCOL_V2_REFERENCE_CSV,
    SEMANTIC_REVIEW_MANIFEST_CSV,
    V2_NEGATIVE,
    V2_POSITIVE,
    read_str_csv,
    validate_reference,
)
from src.paths import OUTPUTS_DIR, RAW_PATCHES_ROOT

MODE_UNMATCHED = "unmatched"
MODE_POSITIVE_QC = "positive-qc"
MODE_CONFIDENCE_PILOT = "confidence-pilot"
MODES = (MODE_UNMATCHED, MODE_POSITIVE_QC, MODE_CONFIDENCE_PILOT)

REVIEW_OUTPUT_ROOT = OUTPUTS_DIR / "semantic_gt_review"
POSITIVE_QC_MANIFEST_CSV = OUTPUTS_DIR / "diagnostics" / "gt_positive_blind_qc" / "blind_manifest.csv"
CANDIDATE_AUDIT_ROOT = OUTPUTS_DIR / "semantic_candidate_audit"
CONFIDENCE_PILOT_MANIFEST_CSV = CANDIDATE_AUDIT_ROOT / "semantic_pilot_manifest.csv"
# Sealed until the blind pilot review is complete; review code must never open it.
SEALED_PILOT_REFERENCE_CSV = CANDIDATE_AUDIT_ROOT / "semantic_pilot_reference.csv"
DEFAULT_OUTPUT = {
    MODE_UNMATCHED: REVIEW_OUTPUT_ROOT / "human_review.csv",
    MODE_POSITIVE_QC: REVIEW_OUTPUT_ROOT / "human_positive_qc.csv",
    MODE_CONFIDENCE_PILOT: REVIEW_OUTPUT_ROOT / "human_confidence_pilot.csv",
}
EXPECTED_COUNT = {MODE_UNMATCHED: EXPECTED_V2_NEGATIVE, MODE_POSITIVE_QC: 100, MODE_CONFIDENCE_PILOT: 400}

LABELS = (PALM, NON_PALM, AMBIGUOUS)
UNLABELED = ""

IDENTITY_COLUMNS = {
    MODE_UNMATCHED: ["review_index", "sample_id", "image_id", "image_path", "yolo_bbox_xywh"],
    MODE_POSITIVE_QC: ["blind_id", "sample_id", "image_id", "image_path", "yolo_bbox_xywh"],
    MODE_CONFIDENCE_PILOT: ["review_index", "pilot_id", "image_id", "image_path", "yolo_bbox_xywh"],
}
DISPLAY_COLUMN = {MODE_UNMATCHED: "sample_id", MODE_POSITIVE_QC: "blind_id", MODE_CONFIDENCE_PILOT: "pilot_id"}
LABEL_COLUMNS = ["semantic_label", "reviewer", "review_timestamp"]

PILOT_ID_PATTERN = re.compile(r"^cp_\d{3}$")
# Fail closed on any column name that looks like ranking, score, GT or geometry metadata.
FORBIDDEN_PILOT_COLUMN = re.compile(
    r"conf|score|bin|iou|match|nearest|geo|gt|protocol|label|prediction|rank|weight|vlm|model|stratum|"
    r"border|overlap|redundan",
    re.IGNORECASE,
)

PROTECTED_OUTPUT_ROOTS = [
    OUTPUTS_DIR / "verification",
    OUTPUTS_DIR / "evaluation",
    OUTPUTS_DIR / "evaluation_protocol_v2",
    OUTPUTS_DIR / "diagnostics",
    OUTPUTS_DIR / "semantic_gt_evaluation",
    OUTPUTS_DIR / "full_inference",
    OUTPUTS_DIR / "verification_dataset",
    CANDIDATE_AUDIT_ROOT,
]

BBOX_TOLERANCE_PX = 1.0
BACKUP_EVERY_CHANGES = 25
BACKUPS_KEPT = 30


class ReviewDataError(ValueError):
    """Startup validation failed; the server must not start."""


class LabelConflict(ValueError):
    """A label write was refused (stale client, or an existing label without explicit change)."""


@dataclass(frozen=True)
class ReviewItem:
    position: int
    display_id: str
    sample_id: str
    image_path: Path
    bbox: tuple[float, float, float, float]
    identity: dict[str, str]


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    values = json.loads(text)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"bbox must be a 4-element list: {text!r}")
    x, y, w, h = (float(v) for v in values)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"bbox must be finite with positive size: {text!r}")
    return x, y, w, h


def _require_columns(frame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ReviewDataError(f"{name} missing columns: {missing}")


def _check_images_and_boxes(rows: list[dict[str, str]], images_root: Path) -> list[str]:
    errors: list[str] = []
    root = images_root.resolve()
    for row in rows:
        key = row.get("sample_id") or row.get("pilot_id")
        path = Path(row["image_path"]).resolve()
        if not path.is_relative_to(root):
            errors.append(f"{key}: image outside {root}: {path}")
            continue
        if not path.is_file():
            errors.append(f"{key}: image not found: {path}")
            continue
        try:
            x, y, w, h = parse_bbox(row["yolo_bbox_xywh"])
        except (ValueError, TypeError) as error:
            errors.append(f"{key}: invalid bbox: {error}")
            continue
        with Image.open(path) as image:
            width, height = image.size
        tol = BBOX_TOLERANCE_PX
        if x < -tol or y < -tol or x + w > width + tol or y + h > height + tol:
            errors.append(f"{key}: bbox {row['yolo_bbox_xywh']} outside image {width}x{height}")
    return errors


def _check_unique(rows: list[dict[str, str]], column: str, errors: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        if row[column] in seen:
            duplicates.add(row[column])
        seen.add(row[column])
    if duplicates:
        errors.append(f"duplicate {column} values: {sorted(duplicates)[:10]}")


def load_unmatched_rows(
    manifest_path: Path = SEMANTIC_REVIEW_MANIFEST_CSV,
    reference_path: Path = PROTOCOL_V2_REFERENCE_CSV,
    expected_count: int = EXPECTED_V2_NEGATIVE,
    reference_counts: tuple[int, int] | None = (EXPECTED_V2_POSITIVE, EXPECTED_V2_NEGATIVE),
) -> list[dict[str, str]]:
    manifest = read_str_csv(manifest_path)
    reference = read_str_csv(reference_path)
    validate_reference(reference, reference_counts)
    _require_columns(manifest, IDENTITY_COLUMNS[MODE_UNMATCHED] + ["original_protocol_v2_gt"], "manifest")

    errors: list[str] = []
    rows = manifest.to_dict("records")
    if len(rows) != expected_count:
        errors.append(f"manifest has {len(rows)} rows, expected {expected_count}")
    _check_unique(rows, "sample_id", errors)
    _check_unique(rows, "review_index", errors)
    try:
        indices = sorted(int(r["review_index"]) for r in rows)
        if indices != list(range(1, len(rows) + 1)):
            errors.append("review_index is not 1..N")
    except ValueError:
        errors.append("review_index is not an integer column")

    not_negative = [r["sample_id"] for r in rows if r["original_protocol_v2_gt"] != V2_NEGATIVE]
    if not_negative:
        errors.append(f"rows that are not Protocol v2 GT-: {not_negative[:10]}")
    ref_negatives = set(reference.loc[reference["original_protocol_v2_gt"] == V2_NEGATIVE, "sample_id"])
    manifest_ids = {r["sample_id"] for r in rows}
    if manifest_ids != ref_negatives:
        errors.append(
            f"manifest sample_ids differ from reference GT- set "
            f"(missing {sorted(ref_negatives - manifest_ids)[:5]}, extra {sorted(manifest_ids - ref_negatives)[:5]})"
        )
    ref_by_id = reference.set_index("sample_id")
    for row in rows:
        if row["sample_id"] not in ref_by_id.index:
            continue
        for column in ("image_id", "image_path", "yolo_bbox_xywh"):
            if row[column] != ref_by_id.at[row["sample_id"], column]:
                errors.append(f"{row['sample_id']}: {column} differs from reference")
    if errors:
        raise ReviewDataError("invalid unmatched review source:\n  " + "\n  ".join(errors[:50]))
    rows.sort(key=lambda r: int(r["review_index"]))
    return [{c: r[c] for c in IDENTITY_COLUMNS[MODE_UNMATCHED]} for r in rows]


def load_positive_qc_rows(
    qc_manifest_path: Path = POSITIVE_QC_MANIFEST_CSV,
    reference_path: Path = PROTOCOL_V2_REFERENCE_CSV,
    expected_count: int = 100,
    reference_counts: tuple[int, int] | None = (EXPECTED_V2_POSITIVE, EXPECTED_V2_NEGATIVE),
) -> list[dict[str, str]]:
    qc = read_str_csv(qc_manifest_path)
    reference = read_str_csv(reference_path)
    validate_reference(reference, reference_counts)
    _require_columns(qc, ["blind_id", "sample_id"], "positive QC manifest")

    errors: list[str] = []
    rows = qc.to_dict("records")
    if len(rows) != expected_count:
        errors.append(f"positive QC manifest has {len(rows)} rows, expected {expected_count}")
    _check_unique(rows, "sample_id", errors)
    _check_unique(rows, "blind_id", errors)
    ref_by_id = reference.set_index("sample_id")
    out: list[dict[str, str]] = []
    for row in rows:
        key = row["sample_id"]
        if key not in ref_by_id.index:
            errors.append(f"{key}: not in Protocol v2 reference")
            continue
        if ref_by_id.at[key, "original_protocol_v2_gt"] != V2_POSITIVE:
            errors.append(f"{key}: not a Protocol v2 GT+ detection")
        out.append({
            "blind_id": row["blind_id"],
            "sample_id": key,
            "image_id": ref_by_id.at[key, "image_id"],
            "image_path": ref_by_id.at[key, "image_path"],
            "yolo_bbox_xywh": ref_by_id.at[key, "yolo_bbox_xywh"],
        })
    if errors:
        raise ReviewDataError("invalid positive QC source:\n  " + "\n  ".join(errors[:50]))
    out.sort(key=lambda r: r["blind_id"])
    return out


def refuse_sealed_reference(path: Path) -> None:
    """Raise unless ``path`` is clearly not the sealed pilot reference."""
    resolved = path.resolve()
    if resolved == SEALED_PILOT_REFERENCE_CSV.resolve() or "reference" in resolved.name.lower():
        raise ReviewDataError(
            f"refusing to read {path}: the pilot reference is sealed until the blind review is complete"
        )


def load_confidence_pilot_rows(
    manifest_path: Path = CONFIDENCE_PILOT_MANIFEST_CSV,
    expected_count: int = EXPECTED_COUNT[MODE_CONFIDENCE_PILOT],
) -> list[dict[str, str]]:
    """Read only the blind pilot manifest, in its stored (already randomized) order."""
    refuse_sealed_reference(manifest_path)
    manifest = read_str_csv(manifest_path)
    expected_columns = IDENTITY_COLUMNS[MODE_CONFIDENCE_PILOT]
    columns = list(manifest.columns)
    forbidden = [c for c in columns if FORBIDDEN_PILOT_COLUMN.search(c)]
    if forbidden:
        raise ReviewDataError(f"blind pilot manifest contains forbidden diagnostic columns: {forbidden}")
    if columns != expected_columns:
        raise ReviewDataError(f"blind pilot manifest columns {columns} != expected {expected_columns}")

    errors: list[str] = []
    rows = manifest.to_dict("records")
    if len(rows) != expected_count:
        errors.append(f"manifest has {len(rows)} rows, expected {expected_count}")
    _check_unique(rows, "pilot_id", errors)
    _check_unique(rows, "review_index", errors)
    seen_targets: set[tuple[str, str]] = set()
    for line, row in enumerate(rows, start=1):
        if row["review_index"] != str(line):
            errors.append(f"line {line}: review_index {row['review_index']!r} (must be 1..N in file order)")
        if not PILOT_ID_PATTERN.match(row["pilot_id"]) or row["pilot_id"] != f"cp_{line:03d}":
            errors.append(f"line {line}: pilot_id {row['pilot_id']!r} is not the neutral cp_{line:03d}")
        if not row["image_id"] or Path(row["image_path"]).stem != row["image_id"]:
            errors.append(f"line {line}: image_id {row['image_id']!r} does not match image_path")
        target = (row["image_path"], row["yolo_bbox_xywh"])
        if target in seen_targets:
            errors.append(f"line {line}: duplicate image/box target")
        seen_targets.add(target)
    if errors:
        raise ReviewDataError("invalid confidence-pilot manifest (not repaired):\n  " + "\n  ".join(errors[:50]))
    return [{c: r[c] for c in expected_columns} for r in rows]


def build_items(mode: str, rows: list[dict[str, str]], images_root: Path = RAW_PATCHES_ROOT) -> list[ReviewItem]:
    errors = _check_images_and_boxes(rows, images_root)
    if errors:
        raise ReviewDataError("image/bbox validation failed:\n  " + "\n  ".join(errors[:50]))
    display_column = DISPLAY_COLUMN[mode]
    return [
        ReviewItem(
            position=position,
            display_id=row[display_column],
            sample_id=row.get("sample_id") or row[display_column],
            image_path=Path(row["image_path"]).resolve(),
            bbox=parse_bbox(row["yolo_bbox_xywh"]),
            identity=dict(row),
        )
        for position, row in enumerate(rows)
    ]


def check_output_path(output_path: Path, source_paths: list[Path], mode: str | None = None) -> Path:
    resolved = output_path.resolve()
    for root in PROTECTED_OUTPUT_ROOTS:
        if resolved.is_relative_to(root.resolve()):
            raise ReviewDataError(f"refusing to write review labels under protected {root}")
    if mode is not None:
        for other_mode, other_output in DEFAULT_OUTPUT.items():
            if other_mode != mode and resolved == other_output.resolve():
                raise ReviewDataError(f"refusing to write {mode} labels into the {other_mode} label file {other_output}")
    for source in source_paths:
        if resolved == source.resolve():
            raise ReviewDataError(f"refusing to write review labels over source manifest {source}")
    return resolved


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


class ReviewStore:
    """Thread-safe label store backed by one CSV, an append-only JSONL log and rotating backups."""

    def __init__(
        self,
        mode: str,
        items: list[ReviewItem],
        output_path: Path,
        reviewer: str,
        backup_every: int = BACKUP_EVERY_CHANGES,
        backups_kept: int = BACKUPS_KEPT,
    ) -> None:
        if not reviewer or not reviewer.strip():
            raise ReviewDataError("reviewer is required")
        self.mode = mode
        self.items = items
        self.output_path = output_path
        self.reviewer = reviewer.strip()
        self.identity_columns = IDENTITY_COLUMNS[mode]
        self.columns = self.identity_columns + LABEL_COLUMNS
        self.log_path = output_path.with_suffix(".log.jsonl")
        self.backup_dir = output_path.parent / "backups"
        self.backup_every = backup_every
        self.backups_kept = backups_kept
        self._lock = threading.Lock()
        self._changes_since_backup = 0
        self.labels = [UNLABELED] * len(items)
        self.reviewers = [""] * len(items)
        self.timestamps = [""] * len(items)
        if output_path.exists():
            self._load_existing()
            self._backup("startup")
        else:
            self._save()

    def _load_existing(self) -> None:
        frame = read_str_csv(self.output_path)
        if list(frame.columns) != self.columns:
            raise ReviewDataError(
                f"{self.output_path}: columns {list(frame.columns)} != expected {self.columns}"
            )
        if len(frame) != len(self.items):
            raise ReviewDataError(f"{self.output_path}: {len(frame)} rows, expected {len(self.items)}")
        errors: list[str] = []
        for position, (item, row) in enumerate(zip(self.items, frame.to_dict("records"))):
            for column in self.identity_columns:
                if row[column] != item.identity[column]:
                    errors.append(f"row {position + 1}: {column} {row[column]!r} != source {item.identity[column]!r}")
            label = row["semantic_label"]
            if label not in (*LABELS, UNLABELED):
                errors.append(f"row {position + 1} ({item.sample_id}): invalid semantic_label {label!r}")
            elif label and not (row["reviewer"] and row["review_timestamp"]):
                errors.append(f"row {position + 1} ({item.sample_id}): label without reviewer/timestamp")
            elif not label and (row["reviewer"] or row["review_timestamp"]):
                errors.append(f"row {position + 1} ({item.sample_id}): reviewer/timestamp without label")
            self.labels[position] = label
            self.reviewers[position] = row["reviewer"]
            self.timestamps[position] = row["review_timestamp"]
        if errors:
            raise ReviewDataError(
                f"{self.output_path} disagrees with the source (not repaired):\n  " + "\n  ".join(errors[:50])
            )

    def _rows(self) -> list[dict[str, str]]:
        return [
            {**item.identity, "semantic_label": label, "reviewer": reviewer, "review_timestamp": stamp}
            for item, label, reviewer, stamp in zip(self.items, self.labels, self.reviewers, self.timestamps)
        ]

    def _save(self) -> None:
        atomic_write_csv(self.output_path, self.columns, self._rows())

    def _backup(self, tag: str) -> None:
        if not self.output_path.exists():
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.backup_dir / f"{self.output_path.stem}.{stamp}.{tag}.csv"
        atomic_write_csv(target, self.columns, self._rows())
        backups = sorted(self.backup_dir.glob(f"{self.output_path.stem}.*.csv"))
        for old in backups[: max(0, len(backups) - self.backups_kept)]:
            old.unlink()
        self._changes_since_backup = 0

    def _log(self, event: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def completed(self) -> int:
        return sum(1 for label in self.labels if label)

    def first_unlabeled(self) -> int | None:
        return next((i for i, label in enumerate(self.labels) if not label), None)

    def next_unlabeled(self, after: int) -> int | None:
        n = len(self.labels)
        for step in range(1, n + 1):
            position = (after + step) % n
            if not self.labels[position]:
                return position
        return None

    def set_label(
        self,
        position: int,
        display_id: str,
        label: str,
        expected_current: str,
        confirm_change: bool = False,
    ) -> dict[str, object]:
        if label not in (*LABELS, UNLABELED):
            raise LabelConflict(f"invalid label {label!r}; allowed {LABELS}")
        with self._lock:
            if not 0 <= position < len(self.items):
                raise LabelConflict(f"position {position} out of range")
            item = self.items[position]
            if item.display_id != display_id:
                raise LabelConflict(f"position {position} is {item.display_id}, not {display_id}")
            current = self.labels[position]
            if expected_current != current:
                raise LabelConflict(f"stale page: current label is {current or 'unlabeled'}; reload")
            if label == current:
                return {"changed": False, "label": current}
            if (current or label == UNLABELED) and not confirm_change:
                raise LabelConflict(f"{item.display_id} already labeled {current}; explicit change required")
            previous = (current, self.reviewers[position], self.timestamps[position])
            self.labels[position] = label
            self.reviewers[position] = self.reviewer if label else ""
            self.timestamps[position] = _utc_now() if label else ""
            try:
                self._save()
            except BaseException:
                self.labels[position], self.reviewers[position], self.timestamps[position] = previous
                raise
            self._log({
                "timestamp": _utc_now(), "reviewer": self.reviewer, "mode": self.mode,
                "position": position, "sample_id": item.sample_id, "display_id": item.display_id,
                "previous": current, "new": label,
            })
            self._changes_since_backup += 1
            if self._changes_since_backup >= self.backup_every:
                self._backup("periodic")
            return {"changed": True, "label": label}
