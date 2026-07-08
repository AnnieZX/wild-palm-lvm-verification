"""Load experiment artifacts for qualitative verification figures."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.paths import (
    ABLATION_CODES,
    ABLATION_CODE_TO_CONDITION,
    PREDICTIONS_FULL_JSON,
    QWEN_EVALUATION_ROOT,
    QWEN_VERIFICATION_ROOT,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_DIR,
    VERIFICATION_DATASET_INDEX_CSV,
    discover_ablation_inputs_dir,
    qwen_evaluation_condition_dir,
    qwen_experiment_visualization_dir,
    qwen_verification_condition_dir,
)

logger = logging.getLogger(__name__)

EVALUATION_CSV_PATTERN = re.compile(r"^(A\d+)_evaluation\.csv$")


@dataclass(frozen=True)
class ExperimentContext:
    """Resolved paths for one Qwen ablation experiment."""

    experiment_id: str
    output_dir: Path
    verification_root: Path
    evaluation_root: Path
    dataset_dir: Path
    dataset_index_csv: Path
    images_root: Path
    annotations_root: Path
    predictions_json: Path
    ablation_inputs_dir: Path
    primary_ablation: str


@dataclass(frozen=True)
class SampleRecord:
    """Merged evaluation and verification fields for one sample."""

    sample_id: str
    image_name: str
    ablation_code: str
    yolo_bbox: tuple[float, float, float, float] | None
    gt_bbox: tuple[float, float, float, float] | None
    max_iou: float
    matched_gt: bool
    yolo_confidence: float | None
    verification_label: str
    decision: str
    confidence_reasoning: str
    visual_reasoning: str
    gt_label: str


def build_experiment_context(
    experiment_id: str,
    output_dir: Path | None = None,
    *,
    dataset_dir: Path | None = None,
    sample_size: int | None = None,
    primary_ablation: str = "A3",
) -> ExperimentContext:
    """Resolve all paths for a Qwen experiment visualization run."""
    if primary_ablation not in ABLATION_CODE_TO_CONDITION:
        raise ValueError(f"Unknown primary ablation {primary_ablation!r}")

    resolved_output = output_dir or qwen_experiment_visualization_dir(experiment_id)
    resolved_dataset = dataset_dir or VERIFICATION_DATASET_DIR
    index_csv = resolved_dataset / "index.csv"
    if not index_csv.exists():
        index_csv = VERIFICATION_DATASET_INDEX_CSV

    return ExperimentContext(
        experiment_id=experiment_id,
        output_dir=resolved_output,
        verification_root=QWEN_VERIFICATION_ROOT / experiment_id,
        evaluation_root=QWEN_EVALUATION_ROOT / experiment_id,
        dataset_dir=resolved_dataset,
        dataset_index_csv=index_csv,
        images_root=RAW_PATCHES_ROOT,
        annotations_root=RAW_PATCHES_ROOT,
        predictions_json=PREDICTIONS_FULL_JSON,
        ablation_inputs_dir=discover_ablation_inputs_dir(sample_size),
        primary_ablation=primary_ablation,
    )


def verification_results_dir(context: ExperimentContext, ablation_code: str) -> Path:
    return qwen_verification_condition_dir(ablation_code, context.experiment_id)


def evaluation_dir_for_ablation(context: ExperimentContext, ablation_code: str) -> Path:
    return qwen_evaluation_condition_dir(ablation_code, context.experiment_id)


def ablation_image_path(
    context: ExperimentContext,
    ablation_code: str,
    sample_id: str,
) -> Path:
    condition_name = ABLATION_CODE_TO_CONDITION[ablation_code]
    return context.ablation_inputs_dir / condition_name / "images" / f"{sample_id}.png"


def overlay_dataset_image_path(context: ExperimentContext, sample_id: str) -> Path:
    return context.dataset_dir / "images" / f"{sample_id}.png"


def load_verification_result(results_dir: Path, sample_id: str) -> dict[str, Any]:
    path = results_dir / f"{sample_id}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def load_evaluation_csv(evaluation_dir: Path) -> pd.DataFrame:
    matches = sorted(
        path
        for path in evaluation_dir.glob("*_evaluation.csv")
        if EVALUATION_CSV_PATTERN.match(path.name)
    )
    if not matches:
        raise FileNotFoundError(f"No evaluation CSV found under {evaluation_dir}")
    return pd.read_csv(matches[0])


def load_ablation_f1(context: ExperimentContext, ablation_code: str) -> float:
    metrics_path = evaluation_dir_for_ablation(context, ablation_code) / f"{ablation_code}_metrics.json"
    if not metrics_path.exists():
        return 0.0
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    return float(metrics.get("f1", 0.0))


def find_best_ablation_code(context: ExperimentContext) -> str:
    scores = {code: load_ablation_f1(context, code) for code in ABLATION_CODES}
    return max(scores, key=scores.get)


def parse_bbox_field(value: Any) -> tuple[float, float, float, float] | None:
    from src.visualization.verification_visualization import parse_bbox_json

    return parse_bbox_json(value)


def row_to_sample_record(row: pd.Series, result: dict[str, Any], ablation_code: str) -> SampleRecord:
    confidence_value = row.get("yolo_confidence")
    try:
        yolo_confidence = (
            float(confidence_value)
            if pd.notna(confidence_value) and str(confidence_value).strip()
            else None
        )
    except (TypeError, ValueError):
        yolo_confidence = None

    matched_gt = str(row.get("matched_gt", "")).strip().lower() in {"true", "1", "yes"}
    verification_label = str(row.get("verification_label", "") or result.get("decision", "") or "").strip()
    decision = str(result.get("decision", verification_label) or "").strip()

    return SampleRecord(
        sample_id=str(row["sample_id"]),
        image_name=str(row["image_name"]),
        ablation_code=ablation_code,
        yolo_bbox=parse_bbox_field(row.get("yolo_bbox")),
        gt_bbox=parse_bbox_field(row.get("gt_bbox")),
        max_iou=float(row.get("max_iou", 0.0) or 0.0),
        matched_gt=matched_gt,
        yolo_confidence=yolo_confidence,
        verification_label=verification_label,
        decision=decision,
        confidence_reasoning=str(result.get("confidence_reasoning", "") or ""),
        visual_reasoning=str(result.get("visual_reasoning", "") or ""),
        gt_label="Positive" if matched_gt else "Negative",
    )


def load_primary_evaluation_pool(context: ExperimentContext) -> pd.DataFrame:
    evaluation_dir = evaluation_dir_for_ablation(context, context.primary_ablation)
    if not evaluation_dir.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {evaluation_dir}. "
            "Run evaluate_verification_against_groundtruth.py first."
        )
    df = load_evaluation_csv(evaluation_dir)
    if df.empty:
        raise ValueError(f"Evaluation CSV is empty under {evaluation_dir}")
    return df


def select_samples(pool: pd.DataFrame, sample_count: int, seed: int) -> pd.DataFrame:
    count = min(sample_count, len(pool))
    return pool.sample(n=count, random_state=seed).reset_index(drop=True)


def classify_failure_case(record: SampleRecord) -> str | None:
    decision = record.decision or record.verification_label
    if decision == "Uncertain":
        return "uncertain"
    if not record.matched_gt and decision == "Reliable":
        return "false_positive"
    if record.matched_gt and decision == "Unreliable":
        return "false_negative"
    return None


def ensure_output_subdirs(output_dir: Path) -> dict[str, Path]:
    subdirs = {
        "overlay": output_dir / "overlay",
        "comparison": output_dir / "comparison",
        "failure_cases": output_dir / "failure_cases",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs
