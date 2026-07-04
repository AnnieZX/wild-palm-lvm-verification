"""Prompt variants for verification input ablation (A1–A4)."""

from __future__ import annotations

from src.prompts.verification_prompt import DECISION_DEFINITIONS, json_response_template

ABLATION_CONDITIONS = (
    "A1_overlay_only",
    "A2_overlay_confidence",
    "A3_overlay_confidence_geometry",
    "A4_overlay_crop_confidence",
)

PALM_DOMAIN_GUIDANCE = """
Wild palms in aerial orthomosaic imagery often show:
- radial or crown-like structure
- long feather-like fronds
- fronds extending from a central crown
- texture distinct from dense broadleaf canopy
- partially occluded palms may still be valid if palm frond structure is visible

Do not require the full crown to be perfectly visible.
A partial but recognizable palm inside the box can be Reliable.
"""

_IMAGE_OVERLAY_ONLY = """
The input image is an orthomosaic patch where:
- the background outside the target region is slightly dimmed
- a green bounding box highlights the candidate detection to review

Focus only on the highlighted bounding box. Determine whether it contains a valid wild palm.
"""

_IMAGE_A4_COMBINED = """
The input image is a two-panel view:
- left panel: the full orthomosaic patch with the highlighted detection (dimmed background, green box)
- right panel: an enlarged crop of the highlighted bounding-box region

Use both panels together. Determine whether the highlighted detection contains a valid wild palm.
"""


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


def _base_prompt(
    sample_id: str,
    condition: str,
    image_description: str,
    metadata_section: str,
    metadata_instruction: str,
) -> str:
    return f"""You are an expert reviewer for aerial wild palm detection in orthomosaic imagery.

Your task is NOT to detect new palms.
Your task is to verify whether a YOLO-generated detection is valid.

Ablation condition: {condition}
Sample ID: {sample_id}

{PALM_DOMAIN_GUIDANCE}

{image_description}

{metadata_section}{metadata_instruction}

{DECISION_DEFINITIONS}

Return ONLY valid JSON with no markdown, no code fences, and no extra text.
Use this exact structure:
{json_response_template()}
"""


def build_ablation_verification_prompt(metadata: dict, condition: str) -> str:
    """
    Build one ablation verification prompt.

    Args:
        metadata: Verification sample metadata dictionary.
        condition: One of A1_overlay_only … A4_overlay_crop_confidence.
    """
    if condition not in ABLATION_CONDITIONS:
        allowed = ", ".join(ABLATION_CONDITIONS)
        raise ValueError(f"Unknown ablation condition {condition!r}. Expected one of: {allowed}")

    sample_id = str(metadata.get("sample_id", "unknown"))

    if condition == "A1_overlay_only":
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=_IMAGE_OVERLAY_ONLY,
            metadata_section="",
            metadata_instruction=(
                "Judge only from the highlighted bounding box in the image.\n"
                "Do not use YOLO confidence or bounding-box geometry."
            ),
        )

    if condition == "A2_overlay_confidence":
        lines = _metadata_lines(metadata, [("confidence", "YOLO confidence")])
        metadata_section = ""
        if lines:
            metadata_section = "Detection metadata:\n" + "\n".join(lines) + "\n"
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=_IMAGE_OVERLAY_ONLY,
            metadata_section=metadata_section,
            metadata_instruction=(
                "Use YOLO confidence only as auxiliary context.\n"
                "Do not use bounding-box width, height, area, center, or coordinates."
            ),
        )

    if condition == "A3_overlay_confidence_geometry":
        aspect = _aspect_ratio(metadata)
        extra: dict[str, float] = {}
        if aspect is not None:
            extra["aspect_ratio"] = round(aspect, 4)

        lines = _metadata_lines(
            metadata,
            [
                ("confidence", "YOLO confidence"),
                ("bbox_width", "Bounding box width (px)"),
                ("bbox_height", "Bounding box height (px)"),
                ("bbox_area", "Bounding box area (px²)"),
            ],
        )
        if extra.get("aspect_ratio") is not None:
            lines.append(f"- Aspect ratio (width/height): {extra['aspect_ratio']}")

        metadata_section = "Detection metadata:\n" + "\n".join(lines) + "\n" if lines else ""
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=_IMAGE_OVERLAY_ONLY,
            metadata_section=metadata_section,
            metadata_instruction=(
                "Use YOLO confidence and the provided geometry as auxiliary context.\n"
                "Do not use center coordinates or raw bbox coordinates."
            ),
        )

    if condition == "A4_overlay_crop_confidence":
        lines = _metadata_lines(metadata, [("confidence", "YOLO confidence")])
        metadata_section = ""
        if lines:
            metadata_section = "Detection metadata:\n" + "\n".join(lines) + "\n"
        return _base_prompt(
            sample_id=sample_id,
            condition=condition,
            image_description=_IMAGE_A4_COMBINED,
            metadata_section=metadata_section,
            metadata_instruction=(
                "Use YOLO confidence only as auxiliary context.\n"
                "Do not use bounding-box geometry beyond what is visible in the panels."
            ),
        )

    raise ValueError(f"Unhandled ablation condition: {condition}")
