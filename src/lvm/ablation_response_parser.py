"""Parse and normalize ablation study model responses."""

from __future__ import annotations

from typing import Any

from src.lvm.response_schema import parse_json_response

DETECTION_QUALITY_VALUES = {"reliable", "uncertain", "unreliable"}
IS_PALM_VALUES = {"yes", "uncertain", "no"}


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_lower(value: Any) -> str:
    return _as_string(value).lower()


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _flatten_visual_evidence(parsed: dict[str, Any]) -> dict[str, str]:
    visual = parsed.get("visual_evidence", {})
    if not isinstance(visual, dict):
        return {
            "radial_crown_visible": "",
            "fronds_visible": "",
            "trunk_visible": "",
        }
    return {
        "radial_crown_visible": _normalize_lower(visual.get("radial_crown_visible")),
        "fronds_visible": _normalize_lower(visual.get("fronds_visible")),
        "trunk_visible": _normalize_lower(visual.get("trunk_visible")),
    }


def _apply_legacy_mappings(parsed: dict[str, Any]) -> dict[str, Any]:
    updated = dict(parsed)

    if not updated.get("detection_quality"):
        for legacy_key in ("classification", "reliability", "label"):
            if updated.get(legacy_key):
                updated["detection_quality"] = updated[legacy_key]
                break

    if updated.get("palm_confidence") in (None, ""):
        if updated.get("confidence") not in (None, ""):
            updated["palm_confidence"] = updated["confidence"]

    return updated


def parse_ablation_response(raw_text: str, prompt_variant: str) -> dict[str, Any]:
    """
    Parse one ablation model response into a flat normalized dictionary.

    Raises ValueError if JSON cannot be parsed.
    """
    parsed = parse_json_response(raw_text)
    parsed = _apply_legacy_mappings(parsed)
    visual = _flatten_visual_evidence(parsed)

    result: dict[str, Any] = {
        "palm_id": _as_string(parsed.get("palm_id")),
        "is_palm": _normalize_lower(parsed.get("is_palm")),
        "palm_confidence": _maybe_float(parsed.get("palm_confidence")),
        "detection_quality": _normalize_lower(parsed.get("detection_quality")),
        "confidence": _maybe_float(parsed.get("confidence")),
        "bbox_alignment": _normalize_lower(parsed.get("bbox_alignment")),
        "palm_structure": _normalize_lower(parsed.get("palm_structure")),
        "occlusion_level": _normalize_lower(parsed.get("occlusion_level")),
        "radial_crown_visible": visual["radial_crown_visible"],
        "fronds_visible": visual["fronds_visible"],
        "trunk_visible": visual["trunk_visible"],
        "reasoning": _as_string(parsed.get("reasoning")),
    }

    if prompt_variant == "P1_direct_reliability":
        if result["detection_quality"] and result["detection_quality"] not in DETECTION_QUALITY_VALUES:
            raise ValueError(
                f"Invalid detection_quality for P1: {result['detection_quality']!r}"
            )
        if result["confidence"] is None:
            raise ValueError("Missing confidence for P1 response.")
        return result

    missing_fields = []
    if _is_missing(result["is_palm"]):
        missing_fields.append("is_palm")
    if _is_missing(result["detection_quality"]):
        missing_fields.append("detection_quality")
    if result["palm_confidence"] is None:
        missing_fields.append("palm_confidence")
    if missing_fields:
        raise ValueError(f"Missing required fields for {prompt_variant}: {missing_fields}")

    if result["is_palm"] and result["is_palm"] not in IS_PALM_VALUES:
        raise ValueError(f"Invalid is_palm value: {result['is_palm']!r}")

    if result["detection_quality"] not in DETECTION_QUALITY_VALUES:
        raise ValueError(f"Invalid detection_quality: {result['detection_quality']!r}")

    return result
