"""Shared verification response parsing pipeline."""

from __future__ import annotations

from typing import Any

from src.lvm.parsers.cleanup import normalize_raw_response
from src.lvm.response_schema import parse_json_response

DECISION_VALUES = ("Reliable", "Uncertain", "Unreliable")

_DECISION_ALIASES = {
    "reliable": "Reliable",
    "uncertain": "Uncertain",
    "unreliable": "Unreliable",
}


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_decision(value: Any) -> str:
    """Normalize decision to Reliable | Uncertain | Unreliable."""
    text = _as_string(value)
    if not text:
        raise ValueError("Missing decision field.")

    if text in DECISION_VALUES:
        return text

    mapped = _DECISION_ALIASES.get(text.lower())
    if mapped:
        return mapped

    allowed = ", ".join(DECISION_VALUES)
    raise ValueError(f"decision must be one of [{allowed}], got {text!r}")


def parse_verification_response(raw_text: str, *, model_key: str = "") -> dict[str, str]:
    """
    Parse and validate one verification model response.

    Pipeline:
        raw model text → model-specific cleanup → JSON parse → schema validation
    """
    cleaned = normalize_raw_response(raw_text, model_key=model_key)
    parsed = parse_json_response(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response must be a JSON object.")

    return {
        "decision": normalize_decision(parsed.get("decision")),
        "confidence_reasoning": _as_string(parsed.get("confidence_reasoning")),
        "visual_reasoning": _as_string(parsed.get("visual_reasoning")),
    }
