"""Mock verifier for testing the LVM pipeline without a real model."""

from __future__ import annotations

from pathlib import Path

from src.lvm.base_verifier import BaseVerifier
from src.lvm.response_schema import validate_lvm_response

# Palms smaller than this area (px²) are treated as uncertain in the mock rules.
SMALL_BBOX_AREA = 10000.0


class MockVerifier(BaseVerifier):
    """
    Placeholder verifier that uses simple heuristics.

    This does not load or call any vision-language model.
    Replace this class with a real open-source VLM implementation later.
    """

    def verify_image(self, image_path: str, metadata: dict) -> dict:
        image_path = Path(image_path)
        bbox_area = float(metadata.get("bbox_area", 0))
        endpoints_count = int(metadata.get("endpoints_count", 0))
        palm_id = metadata.get("palm_id", "unknown")

        if bbox_area < SMALL_BBOX_AREA:
            classification = "uncertain"
            confidence = 0.45
            bbox_alignment = "partial"
            palm_structure = "weak"
            occlusion_level = "medium"
            reasoning = (
                f"Mock result: small bbox area ({bbox_area:.0f} px²) "
                f"below {SMALL_BBOX_AREA:.0f} px²."
            )
        elif endpoints_count >= 3:
            classification = "reliable"
            confidence = 0.85
            bbox_alignment = "good"
            palm_structure = "clear"
            occlusion_level = "low"
            reasoning = (
                f"Mock result: {endpoints_count} endpoints detected (>= 3)."
            )
        else:
            classification = "uncertain"
            confidence = 0.55
            bbox_alignment = "partial"
            palm_structure = "partial"
            occlusion_level = "medium"
            reasoning = (
                f"Mock result: only {endpoints_count} endpoint(s) detected (< 3)."
            )

        raw_response = {
            "palm_id": palm_id,
            "classification": classification,
            "confidence": confidence,
            "bbox_alignment": bbox_alignment,
            "palm_structure": palm_structure,
            "occlusion_level": occlusion_level,
            "reasoning": reasoning,
        }
        validated = validate_lvm_response(raw_response)

        return {
            "image_name": metadata.get("image_name", image_path.name),
            "palm_id": validated["palm_id"],
            "classification": validated["classification"],
            "confidence": validated["confidence"],
            "bbox_alignment": validated["bbox_alignment"],
            "palm_structure": validated["palm_structure"],
            "occlusion_level": validated["occlusion_level"],
            "reasoning": validated["reasoning"],
            "model_name": self.model_name,
        }
