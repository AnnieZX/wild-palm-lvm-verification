"""JSON response schema utilities for LVM verification."""

from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_CLASSIFICATIONS = {"reliable", "uncertain", "unreliable"}

REQUIRED_FIELDS = {
    "palm_id",
    "classification",
    "confidence",
    "bbox_alignment",
    "palm_structure",
    "occlusion_level",
    "reasoning",
}

OPTIONAL_FIELD_DEFAULTS = {
    "bbox_alignment": "unknown",
    "palm_structure": "unknown",
    "occlusion_level": "unknown",
}


def parse_json_response(raw_text: str) -> dict:
    """
    Parse JSON from raw model text.

    Handles markdown code fences and extra text before or after the JSON object.

    Raises:
        ValueError: If no valid JSON object can be parsed.
    """
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty response text; expected JSON object.")

    # Try direct parse first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Remove markdown code fences such as ```json ... ```
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fall back to the first JSON object found in the text.
    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        try:
            parsed = json.loads(object_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as error:
            raise ValueError(f"Could not parse JSON from model response: {error}") from error

    raise ValueError("Could not find a valid JSON object in model response.")


def _normalize_confidence(value: Any) -> float:
    """Convert confidence to a float between 0 and 1."""
    try:
        confidence = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"confidence must be a number, got {value!r}") from error

    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    return confidence


def validate_lvm_response(response: dict) -> dict:
    """
    Validate and clean one LVM verification response.

    Returns:
        Cleaned dictionary with required schema fields.
    """
    if not isinstance(response, dict):
        raise ValueError("Response must be a dictionary.")

    cleaned: dict[str, Any] = {}

    for field in REQUIRED_FIELDS:
        if field not in response or response[field] is None:
            if field in OPTIONAL_FIELD_DEFAULTS:
                cleaned[field] = OPTIONAL_FIELD_DEFAULTS[field]
            elif field == "reasoning":
                cleaned[field] = ""
            else:
                raise ValueError(f"Missing required field: {field}")
        else:
            cleaned[field] = response[field]

    cleaned["classification"] = str(cleaned["classification"]).strip().lower()
    if cleaned["classification"] not in REQUIRED_CLASSIFICATIONS:
        allowed = ", ".join(sorted(REQUIRED_CLASSIFICATIONS))
        raise ValueError(
            f"classification must be one of [{allowed}], got {cleaned['classification']!r}"
        )

    cleaned["confidence"] = _normalize_confidence(cleaned["confidence"])
    cleaned["palm_id"] = str(cleaned["palm_id"])
    cleaned["reasoning"] = str(cleaned["reasoning"]).strip()

    for field, default in OPTIONAL_FIELD_DEFAULTS.items():
        value = cleaned.get(field)
        if value is None or str(value).strip() == "":
            cleaned[field] = default
        else:
            cleaned[field] = str(value).strip().lower()

    return cleaned
