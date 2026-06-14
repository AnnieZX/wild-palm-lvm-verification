"""Prompt templates for LVM palm verification."""

from __future__ import annotations

# Metadata keys that can be included in the prompt when present.
OPTIONAL_METADATA_FIELDS = [
    ("yolo_confidence", "YOLO confidence"),
    ("bbox_area", "Bounding box area (px²)"),
    ("bbox_width", "Bounding box width (px)"),
    ("bbox_height", "Bounding box height (px)"),
    ("endpoints_count", "Endpoint count"),
]


def _format_metadata_lines(palm_metadata: dict) -> list[str]:
    """Build metadata lines for fields that exist and are not empty."""
    lines: list[str] = []
    for key, label in OPTIONAL_METADATA_FIELDS:
        value = palm_metadata.get(key)
        if value is None or value == "":
            continue
        lines.append(f"- {label}: {value}")
    return lines


def build_verification_prompt(palm_metadata: dict) -> str:
    """
    Build a verification prompt for one annotated palm instance.

    Args:
        palm_metadata: Dictionary with at least palm_id and optional fields
            such as yolo_confidence, bbox_area, bbox_width, bbox_height,
            and endpoints_count.

    Returns:
        Prompt string instructing the model to return only valid JSON.
    """
    palm_id = palm_metadata.get("palm_id", "unknown")
    metadata_lines = _format_metadata_lines(palm_metadata)

    metadata_section = ""
    if metadata_lines:
        metadata_section = (
            "\nAdditional detection metadata:\n"
            + "\n".join(metadata_lines)
            + "\n"
        )

    return f"""You are an expert reviewer for aerial wild palm detection in orthomosaic imagery.

Your task is NOT to detect new palms.
Your task is to verify whether the highlighted/annotated palm instance is trustworthy.

The input image shows one candidate palm with visual annotations:
- a bounding box
- a center point
- endpoint markers
- a visible palm ID label

Palm ID: {palm_id}
{metadata_section}
Classify this annotation using exactly one of these labels:

Reliable:
The annotation clearly corresponds to a real palm.

Uncertain:
The annotation may correspond to a palm, but evidence is ambiguous due to occlusion,
overlap, image quality, partial visibility, or unclear structure.

Unreliable:
The annotation is likely incorrect, poorly aligned, or does not correspond to a palm.

Return ONLY valid JSON with no extra text. Use this exact structure:
{{
  "palm_id": "{palm_id}",
  "classification": "reliable | uncertain | unreliable",
  "confidence": 0.0,
  "bbox_alignment": "good | partial | poor | unknown",
  "palm_structure": "clear | partial | weak | absent | unknown",
  "occlusion_level": "low | medium | high | unknown",
  "reasoning": "one short sentence"
}}
"""
