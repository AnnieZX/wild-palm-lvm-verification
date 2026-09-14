"""Shared constants for filesystem experiment discovery (no src/ imports)."""

from __future__ import annotations

from typing import Dict

from shared.models import AblationCode

ABLATION_CONDITION_NAMES: Dict[AblationCode, str] = {
    AblationCode.A1: "A1_overlay_only",
    AblationCode.A2: "A2_overlay_confidence",
    AblationCode.A3: "A3_overlay_confidence_geometry",
    AblationCode.A4: "A4_overlay_crop_confidence",
    AblationCode.A5: "A5_crop_only",
}

ABLATION_CODE_TO_CONDITION: Dict[str, str] = {
    code.value: name for code, name in ABLATION_CONDITION_NAMES.items()
}

ALL_ABLATION_CODES = tuple(AblationCode)

# Directory names under outputs/verification/ mapped to API model_key values.
LEGACY_MODEL_DIR_ALIASES: Dict[str, str] = {
    "qwen": "qwen2_5_vl",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen2_5_vl": "Qwen2.5-VL",
    "llava": "LLaVA-OneVision",
    "gemma": "Gemma 3",
    "internvl": "InternVL",
    "internvl3": "InternVL3",
    "glm_4_6v_flash": "GLM-4.6V-Flash",
}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "qwen2_5_vl": "Primary production VLM for wild palm verification ablations.",
    "llava": "Secondary VLM adapter for comparative ablation studies.",
    "gemma": "Gemma 3 comparative verification runs.",
    "internvl3": "InternVL3 qualification and comparative runs.",
    "glm_4_6v_flash": "GLM-4.6V-Flash verification and qualification runs.",
}

EVALUATION_CSV_SUFFIX = "_evaluation.csv"
METRICS_JSON_SUFFIX = "_metrics.json"
