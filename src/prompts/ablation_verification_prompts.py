"""Prompt variants for verification input ablation (A1–A5)."""

from __future__ import annotations

ABLATION_CONDITIONS = (
    "A1_overlay_only",
    "A2_overlay_confidence",
    "A3_overlay_confidence_geometry",
    "A4_overlay_crop_confidence",
    "A5_crop_only",
)

JSON_RESPONSE_TEMPLATE = """{
  "decision": "Reliable | Uncertain | Unreliable",
  "confidence_reasoning": "",
  "visual_reasoning": ""
}"""

ROLE_SECTION = """You are an expert reviewer for aerial wild palm detection in UAV orthomosaic imagery.

Your task is NOT to detect new palms.

Your task is to verify whether the highlighted YOLO detection contains a valid wild palm."""

IMPORTANT_INSTRUCTIONS = """Important instructions

The highlighted green bounding box indicates the candidate detection.

Focus primarily on the highlighted object.

Use surrounding context only if it provides useful evidence.

The YOLO confidence score is auxiliary information only.

Do NOT base your decision primarily on the confidence score.

Use visual evidence as the primary basis for your judgment."""

PALM_CHARACTERISTICS = """Wild palm characteristics

Wild palms in aerial orthomosaic imagery often exhibit:

• a visible central crown
• radial or circular crown structure
• long feather-like fronds extending outward
• repeated symmetric leaf patterns
• texture distinguishable from surrounding broadleaf canopy
• fronds radiating from a central point

A palm may be partially occluded.

A partially visible palm can still be considered Reliable if recognizable palm morphology is present.

Dense vegetation without recognizable palm morphology should NOT be considered a valid palm."""

DECISION_DEFINITIONS = """Decision definitions

Reliable
The highlighted bounding box clearly contains a valid wild palm.

Uncertain
Evidence is ambiguous due to occlusion, overlap, low image quality, partial visibility, or confusing vegetation.

Unreliable
The highlighted object is not a palm or the detection is clearly incorrect."""

REASONING_REQUIREMENTS = """Reasoning requirements

Base your reasoning primarily on:

• crown structure
• frond pattern
• texture
• morphology
• spatial arrangement

Do not rely mainly on confidence score."""

_IMAGE_OVERLAY = """Input image
The input is an orthomosaic patch where the background outside the target region is slightly dimmed and a green bounding box highlights the candidate detection."""

_IMAGE_A4_COMBINED = """Input image
The input is a two-panel image:
• left panel: full orthomosaic patch with the highlighted detection (dimmed background, green bounding box)
• right panel: enlarged crop of the highlighted bounding-box region

Use both panels together when judging the detection."""

_IMAGE_A5_CROP_ONLY = """Input image
The input is an enlarged crop of the candidate detection region only.

There is no surrounding full-image context.

Judge whether this cropped region contains a valid wild palm."""


def _metadata_lines(metadata: dict, keys: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for key, label in keys:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        lines.append(f"- {label}: {value}")
    return lines


def _aspect_ratio(metadata: dict) -> float | None:
    width = metadata.get("bbox_width")
    height = metadata.get("bbox_height")
    if width is None or height is None:
        bbox = metadata.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            width, height = bbox[2], bbox[3]
    try:
        width_f = float(width)
        height_f = float(height)
    except (TypeError, ValueError):
        return None
    if height_f <= 0:
        return None
    return width_f / height_f


def _format_metadata_section(lines: list[str]) -> str:
    if not lines:
        return ""
    return "Detection metadata:\n" + "\n".join(lines) + "\n\n"


def _base_prompt(
    sample_id: str,
    condition: str,
    image_description: str,
    metadata_section: str,
    metadata_instruction: str,
) -> str:
    parts = [
        ROLE_SECTION,
        "",
        "--------------------------------------------------",
        "",
        IMPORTANT_INSTRUCTIONS,
        "",
        "--------------------------------------------------",
        "",
        PALM_CHARACTERISTICS,
        "",
        "--------------------------------------------------",
        "",
        image_description,
        "",
        "--------------------------------------------------",
        "",
        DECISION_DEFINITIONS,
        "",
        "--------------------------------------------------",
        "",
        REASONING_REQUIREMENTS,
    ]

    if metadata_section or metadata_instruction:
        parts.extend(
            [
                "",
                "--------------------------------------------------",
                "",
            ]
        )
        if metadata_section:
            parts.append(metadata_section.rstrip())
            parts.append("")
        if metadata_instruction:
            parts.append(metadata_instruction)

    parts.extend(
        [
            "",
            "--------------------------------------------------",
            "",
            f"Ablation condition: {condition}",
            f"Sample ID: {sample_id}",
            "",
            "Return ONLY valid JSON.",
            JSON_RESPONSE_TEMPLATE,
        ]
    )

    return "\n".join(parts)


def build_ablation_verification_prompt(metadata: dict, condition: str) -> str:
    """
    Build one ablation verification prompt.

    Args:
        metadata: Verification sample metadata dictionary.
        condition: One of A1_overlay_only … A5_crop_only.
    """
    if condition not in ABLATION_CONDITIONS:
        allowed = ", ".join(ABLATION_CONDITIONS)
        raise ValueError(f"Unknown ablation condition {condition!r}. Expected one of: {allowed}")

    sample_id = str(metadata.get("sample_id", "unknown"))

    if condition == "A1_overlay_only":
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=_IMAGE_OVERLAY,
            metadata_section="",
            metadata_instruction=(
                "Judge only from the highlighted bounding box in the image.\n"
                "Do not use YOLO confidence or bounding-box geometry."
            ),
        )

    if condition == "A2_overlay_confidence":
        metadata_section = _format_metadata_section(
            _metadata_lines(metadata, [("confidence", "YOLO confidence")])
        )
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=_IMAGE_OVERLAY,
            metadata_section=metadata_section,
            metadata_instruction=(
                "Use YOLO confidence only as auxiliary context.\n"
                "Do not use bounding-box width, height, area, center, or coordinates."
            ),
        )

    if condition == "A3_overlay_confidence_geometry":
        lines = _metadata_lines(
            metadata,
            [
                ("confidence", "YOLO confidence"),
                ("bbox_width", "Bounding box width (px)"),
                ("bbox_height", "Bounding box height (px)"),
                ("bbox_area", "Bounding box area (px²)"),
            ],
        )
        aspect = _aspect_ratio(metadata)
        if aspect is not None:
            lines.append(f"- Aspect ratio (width/height): {round(aspect, 4)}")

        metadata_section = _format_metadata_section(lines)
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=_IMAGE_OVERLAY,
            metadata_section=metadata_section,
            metadata_instruction=(
                "Use YOLO confidence and the provided geometry as auxiliary context.\n"
                "Do not use center coordinates or raw bbox coordinates."
            ),
        )

    if condition in {"A4_overlay_crop_confidence", "A5_crop_only"}:
        metadata_section = _format_metadata_section(
            _metadata_lines(metadata, [("confidence", "YOLO confidence")])
        )
        image_description = (
            _IMAGE_A4_COMBINED if condition == "A4_overlay_crop_confidence" else _IMAGE_A5_CROP_ONLY
        )
        metadata_instruction = (
            "Use YOLO confidence only as auxiliary context.\n"
            "Do not use bounding-box geometry beyond what is visible in the image."
        )
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=image_description,
            metadata_section=metadata_section,
            metadata_instruction=metadata_instruction,
        )

    raise ValueError(f"Unhandled ablation condition: {condition}")
