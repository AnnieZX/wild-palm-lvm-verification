"""Prompt variants for the 100-palm ablation study."""

from __future__ import annotations

PROMPT_VARIANTS = (
    "P1_direct_reliability",
    "P2_two_step",
    "P3_reasoning",
    "P4_yolo_confidence",
    "P5_geometric_metadata",
    "P6_full_metadata",
)

DETECTION_QUALITY_DEFINITIONS = """
detection_quality definitions:
- reliable: object appears to be a palm and annotation aligns well.
- uncertain: object may be a palm OR annotation quality is difficult to assess.
- unreliable: object is not a palm OR annotation alignment is clearly incorrect.
"""

IS_PALM_DEFINITION = """
is_palm definitions (use lowercase):
- yes: the highlighted object appears to be a palm.
- uncertain: palm presence is ambiguous.
- no: the highlighted object does not appear to be a palm.
"""


def _format_value(value) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def _metadata_lines(palm_metadata: dict, keys: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for key, label in keys:
        value = _format_value(palm_metadata.get(key))
        if value:
            lines.append(f"- {label}: {value}")
    return lines


def _p1_json_template(palm_id: str) -> str:
    return f"""{{
  "palm_id": "{palm_id}",
  "detection_quality": "reliable | uncertain | unreliable",
  "confidence": 0.0,
  "reasoning": "one short sentence"
}}"""


def _p2_json_template(palm_id: str) -> str:
    return f"""{{
  "palm_id": "{palm_id}",
  "is_palm": "yes | uncertain | no",
  "palm_confidence": 0.0,
  "detection_quality": "reliable | uncertain | unreliable",
  "bbox_alignment": "good | partial | poor | unknown",
  "palm_structure": "clear | partial | weak | absent | unknown",
  "occlusion_level": "low | medium | high | unknown",
  "visual_evidence": {{
    "radial_crown_visible": "yes | partial | no | unknown",
    "fronds_visible": "yes | partial | no | unknown",
    "trunk_visible": "yes | partial | no | unknown"
  }},
  "reasoning": "one short sentence"
}}"""


def _base_task_text(palm_id: str, metadata_section: str, extra_instructions: str = "") -> str:
    return f"""You are an expert reviewer for aerial wild palm detection in orthomosaic imagery.

Your task is NOT to detect new palms.
Your task is to verify whether a YOLO-generated palm detection is trustworthy.

Do NOT assume the YOLO detection is correct.
Separate palm existence from annotation quality.

The input image shows one candidate palm region.
Palm ID: {palm_id}
{metadata_section}
{extra_instructions}
{DETECTION_QUALITY_DEFINITIONS}
{IS_PALM_DEFINITION}

Return ONLY valid JSON with lowercase enum values. Use this exact structure:
"""


def build_ablation_prompt(palm_metadata: dict, prompt_variant: str) -> str:
    """
    Build a verification prompt for one ablation condition.

    Args:
        palm_metadata: Palm metadata dictionary.
        prompt_variant: One of P1_direct_reliability ... P6_full_metadata.
    """
    if prompt_variant not in PROMPT_VARIANTS:
        allowed = ", ".join(PROMPT_VARIANTS)
        raise ValueError(f"Unknown prompt variant {prompt_variant!r}. Expected one of: {allowed}")

    palm_id = palm_metadata.get("palm_id", "unknown")

    yolo_keys = [("yolo_confidence", "YOLO confidence")]
    geometric_keys = [
        ("bbox_area", "Bounding box area (px²)"),
        ("bbox_width", "Bounding box width (px)"),
        ("bbox_height", "Bounding box height (px)"),
        ("endpoints_count", "Endpoint count"),
    ]

    if prompt_variant == "P1_direct_reliability":
        metadata_section = ""
        extra = "Ask only for detection reliability of the highlighted detection."
        return _base_task_text(palm_id, metadata_section, extra) + _p1_json_template(palm_id)

    metadata_keys: list[tuple[str, str]] = []
    extra = ""

    if prompt_variant == "P2_two_step":
        extra = (
            "First assess whether the highlighted object is a palm.\n"
            "Then assess detection quality and annotation alignment."
        )
    elif prompt_variant == "P3_reasoning":
        extra = (
            "First assess whether the highlighted object is a palm.\n"
            "Then assess detection quality and annotation alignment.\n"
            "Reason step-by-step internally, but return ONLY valid JSON.\n"
            "Keep reasoning to one concise sentence."
        )
    elif prompt_variant == "P4_yolo_confidence":
        metadata_keys = yolo_keys
        extra = (
            "First assess whether the highlighted object is a palm.\n"
            "Then assess detection quality and annotation alignment.\n"
            "Use the provided YOLO confidence only as auxiliary metadata."
        )
    elif prompt_variant == "P5_geometric_metadata":
        metadata_keys = geometric_keys
        extra = (
            "First assess whether the highlighted object is a palm.\n"
            "Then assess detection quality and annotation alignment.\n"
            "Use the provided geometric metadata only as auxiliary context."
        )
    elif prompt_variant == "P6_full_metadata":
        metadata_keys = yolo_keys + geometric_keys
        extra = (
            "First assess whether the highlighted object is a palm.\n"
            "Then assess detection quality and annotation alignment.\n"
            "Use all provided metadata only as auxiliary context."
        )

    metadata_lines = _metadata_lines(palm_metadata, metadata_keys)
    metadata_section = ""
    if metadata_lines:
        metadata_section = "Additional metadata:\n" + "\n".join(metadata_lines) + "\n"

    return _base_task_text(palm_id, metadata_section, extra) + _p2_json_template(palm_id)
