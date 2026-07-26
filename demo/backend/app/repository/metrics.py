"""Verification metrics computed from evaluation CSV rows (stdlib only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.models import (
    AblationCode,
    AblationStatistics,
    ConfusionCounts,
    DecisionDistribution,
)

from app.repository.constants import ABLATION_CONDITION_NAMES

POSITIVE_LABEL = "Reliable"
UNCERTAIN_LABEL = "Uncertain"
NEGATIVE_LABEL = "Unreliable"


def normalize_verification_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for label in (POSITIVE_LABEL, UNCERTAIN_LABEL, NEGATIVE_LABEL):
        if text.lower() == label.lower():
            return label
    return ""


def normalize_matched_gt(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _metric_int(metrics: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _metric_float(metrics: Dict[str, Any], *keys: str, default: float = 0.0) -> Optional[float]:
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def normalize_metrics_payload(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metrics from evaluation CSV computation or precomputed JSON.

    Precomputed files from scripts/compute_verification_metrics.py use
    ground_truth_positive / ground_truth_negative instead of matched_gt_count.
    """
    dataset_size = _metric_int(metrics, "dataset_size", "samples", default=0)
    evaluated_samples = _metric_int(metrics, "evaluated_samples", default=dataset_size)

    matched_gt_count = _metric_int(metrics, "matched_gt_count", "ground_truth_positive")
    unmatched_gt_count = _metric_int(metrics, "unmatched_gt_count", "ground_truth_negative")
    if matched_gt_count == 0 and unmatched_gt_count == 0 and dataset_size > 0:
        matched_gt_count = _metric_int(metrics, "ground_truth_positive")
        unmatched_gt_count = max(dataset_size - matched_gt_count, 0)
    elif unmatched_gt_count == 0 and matched_gt_count > 0 and dataset_size > matched_gt_count:
        unmatched_gt_count = dataset_size - matched_gt_count

    precision = _metric_float(metrics, "precision")
    recall = _metric_float(metrics, "recall")
    f1 = _metric_float(metrics, "f1")
    accuracy = _metric_float(metrics, "accuracy")

    return {
        "dataset_size": dataset_size,
        "evaluated_samples": evaluated_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "average_iou": _metric_float(metrics, "average_iou", default=0.0),
        "average_confidence": _metric_float(metrics, "average_confidence", default=0.0),
        "matched_gt_count": matched_gt_count,
        "unmatched_gt_count": unmatched_gt_count,
        "reliable_pct": _metric_float(metrics, "reliable_pct", default=0.0),
        "uncertain_pct": _metric_float(metrics, "uncertain_pct", default=0.0),
        "unreliable_pct": _metric_float(metrics, "unreliable_pct", default=0.0),
        "true_positive": _metric_int(metrics, "true_positive"),
        "false_positive": _metric_int(metrics, "false_positive"),
        "true_negative": _metric_int(metrics, "true_negative"),
        "false_negative": _metric_int(metrics, "false_negative"),
        "uncertain_predictions": _metric_int(metrics, "uncertain_predictions", "uncertain_count"),
    }


def compute_metrics(rows: List[Dict[str, str]], ablation: str) -> Dict[str, Any]:
    dataset_size = len(rows)
    labels = [normalize_verification_label(row.get("verification_label")) for row in rows]
    gt_positive = [normalize_matched_gt(row.get("matched_gt")) for row in rows]
    evaluated_samples = sum(1 for label in labels if label)

    iou_values = [safe_float(row.get("max_iou")) for row in rows]
    confidence_values = [safe_float(row.get("yolo_confidence")) for row in rows]
    valid_iou = [value for value in iou_values if value is not None]
    valid_confidence = [value for value in confidence_values if value is not None]

    reliable_count = sum(1 for label in labels if label == POSITIVE_LABEL)
    uncertain_count = sum(1 for label in labels if label == UNCERTAIN_LABEL)
    unreliable_count = sum(1 for label in labels if label == NEGATIVE_LABEL)

    tp = fp = fn = tn = 0
    for label, matched in zip(labels, gt_positive):
        if label not in (POSITIVE_LABEL, NEGATIVE_LABEL):
            continue
        predicted_positive = label == POSITIVE_LABEL
        predicted_negative = label == NEGATIVE_LABEL
        actual_positive = matched
        actual_negative = not matched
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_negative and predicted_positive:
            fp += 1
        elif actual_positive and predicted_negative:
            fn += 1
        elif actual_negative and predicted_negative:
            tn += 1

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)

    def pct(count: int) -> float:
        if dataset_size == 0:
            return 0.0
        return round(100.0 * count / dataset_size, 2)

    ground_truth_positive = sum(1 for matched in gt_positive if matched)
    ground_truth_negative = dataset_size - ground_truth_positive

    return {
        "ablation": ablation,
        "samples": dataset_size,
        "dataset_size": dataset_size,
        "evaluated_samples": evaluated_samples,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "uncertain_predictions": uncertain_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "average_iou": round(sum(valid_iou) / len(valid_iou), 4) if valid_iou else 0.0,
        "average_confidence": round(sum(valid_confidence) / len(valid_confidence), 4)
        if valid_confidence
        else 0.0,
        "reliable_count": reliable_count,
        "uncertain_count": uncertain_count,
        "unreliable_count": unreliable_count,
        "reliable_pct": pct(reliable_count),
        "uncertain_pct": pct(uncertain_count),
        "unreliable_pct": pct(unreliable_count),
        "ground_truth_positive": ground_truth_positive,
        "ground_truth_negative": ground_truth_negative,
        "matched_gt_count": ground_truth_positive,
        "unmatched_gt_count": ground_truth_negative,
    }


def metrics_to_ablation_statistics(
    metrics: Dict[str, Any],
    ablation: AblationCode,
) -> AblationStatistics:
    normalized = normalize_metrics_payload(metrics)

    return AblationStatistics(
        ablation=ablation,
        ablation_condition=ABLATION_CONDITION_NAMES[ablation],
        dataset_size=normalized["dataset_size"],
        evaluated_samples=normalized["evaluated_samples"],
        precision=normalized["precision"],
        recall=normalized["recall"],
        f1=normalized["f1"],
        accuracy=normalized["accuracy"],
        average_iou=normalized["average_iou"],
        average_confidence=normalized["average_confidence"],
        matched_gt_count=normalized["matched_gt_count"],
        unmatched_gt_count=normalized["unmatched_gt_count"],
        decision_distribution=DecisionDistribution(
            reliable_pct=normalized["reliable_pct"],
            uncertain_pct=normalized["uncertain_pct"],
            unreliable_pct=normalized["unreliable_pct"],
        ),
        confusion_counts=ConfusionCounts(
            true_positive=normalized["true_positive"],
            false_positive=normalized["false_positive"],
            true_negative=normalized["true_negative"],
            false_negative=normalized["false_negative"],
            uncertain=normalized["uncertain_predictions"],
        ),
    )
