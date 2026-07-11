"""Verification result record schema (model-independent)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_result_record(
    sample_id: str,
    raw_response: str,
    parsed: dict[str, str] | None,
    runtime_seconds: float,
    parse_error: str = "",
    inference_error: str = "",
    *,
    model_key: str = "",
    model_name: str = "",
    condition: str = "",
    experiment_id: str = "",
    timestamp: str = "",
) -> dict[str, Any]:
    """
    Build one verification result JSON record.

    Evaluation scripts read only ``decision``; optional metadata fields are
    for auditability and cross-model provenance.
    """
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "decision": parsed.get("decision", "") if parsed else "",
        "confidence_reasoning": parsed.get("confidence_reasoning", "") if parsed else "",
        "visual_reasoning": parsed.get("visual_reasoning", "") if parsed else "",
        "runtime_seconds": round(runtime_seconds, 4),
        "parse_error": parse_error,
        "inference_error": inference_error,
    }

    if model_key:
        record["model_key"] = model_key
    if model_name:
        record["model_name"] = model_name
    if condition:
        record["condition"] = condition
    if experiment_id:
        record["experiment_id"] = experiment_id
    record["timestamp"] = timestamp or datetime.now(timezone.utc).isoformat()

    return record
