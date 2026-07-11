"""Verification result record schema (model-independent)."""

from __future__ import annotations

from typing import Any


def build_result_record(
    sample_id: str,
    raw_response: str,
    parsed: dict[str, str] | None,
    runtime_seconds: float,
    parse_error: str = "",
    inference_error: str = "",
) -> dict[str, Any]:
    """Build one verification result JSON record."""
    return {
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
