"""Prompt templates for YOLO detection verification with Qwen2.5-VL."""

from __future__ import annotations

DECISION_OPTIONS = ("Reliable", "Uncertain", "Unreliable")

DECISION_DEFINITIONS = """
decision definitions (use exact capitalization):
- Reliable: the highlighted bounding box clearly contains a valid wild palm.
- Uncertain: the box may contain a palm, but evidence is ambiguous due to occlusion,
  overlap, image quality, partial visibility, or unclear crown structure.
- Unreliable: the box does not contain a valid palm, or the detection is clearly incorrect.
"""


def _format_metadata_lines(metadata: dict) -> list[str]:
    """Build auxiliary metadata lines from verification sample JSON."""
    lines: list[str] = []

    field_labels = (
        ("sample_id", "Sample ID"),
        ("image_name", "Source patch"),
        ("confidence", "YOLO confidence"),
        ("bbox_width", "Bounding box width (px)"),
        ("bbox_height", "Bounding box height (px)"),
        ("bbox_area", "Bounding box area (px²)"),
        ("center_x", "Box center x (px)"),
        ("center_y", "Box center y (px)"),
    )
    for key, label in field_labels:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        lines.append(f"- {label}: {value}")

    bbox = metadata.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, width, height = bbox
        lines.append(f"- Bounding box (x, y, w, h): ({x}, {y}, {width}, {height})")

    return lines


def json_response_template() -> str:
    """Return the exact JSON schema the model must produce."""
    return """{
  "decision": "Reliable | Uncertain | Unreliable",
  "confidence_reasoning": "brief explanation of decision confidence",
  "visual_reasoning": "brief explanation of visible palm evidence in the image"
}"""


def build_verification_detection_prompt(metadata: dict) -> str:
    """
    Build a verification prompt for one YOLO detection overlay.

    Args:
        metadata: Verification sample metadata (from metadata/sample_XXXXXX.json).

    Returns:
        Prompt string instructing Qwen2.5-VL to return only valid JSON.
    """
    sample_id = metadata.get("sample_id", "unknown")
    metadata_lines = _format_metadata_lines(metadata)
    metadata_section = ""
    if metadata_lines:
        metadata_section = "Detection metadata:\n" + "\n".join(metadata_lines) + "\n"

    return f"""You are an expert reviewer for aerial wild palm detection in orthomosaic imagery.

Your task is NOT to detect new palms.
Your task is to verify whether a YOLO-generated detection is valid.

The input image is an orthomosaic patch where:
- the background outside the target region is slightly dimmed
- a green bounding box highlights the candidate detection to review

Focus only on the highlighted bounding box. Determine whether it contains a valid wild palm.

Sample ID: {sample_id}
{metadata_section}
Use the YOLO confidence and geometry only as auxiliary context. Do not assume the detection is correct.

{DECISION_DEFINITIONS}

Return ONLY valid JSON with no markdown, no code fences, and no extra text.
Use this exact structure:
{json_response_template()}
"""


def prompt_filename_for_sample(sample_id: str) -> str:
    """Map a verification sample_id to a prompt filename."""
    return f"{sample_id}.txt"
