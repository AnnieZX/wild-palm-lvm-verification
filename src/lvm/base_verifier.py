"""Abstract interface for large vision model verifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    """
    Base class for LVM-based palm verification.

    Subclasses should implement verify_image() and can later wrap
    open-source models such as LLaVA, Qwen-VL, or GeoChat.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def verify_image(self, image_path: str, metadata: dict) -> dict:
        """
        Verify one palm crop/overlay image.

        Args:
            image_path: Path to the LVM input image.
            metadata: Palm metadata such as bbox size and endpoint count.

        Returns:
            Dictionary with keys:
            image_name, palm_id, classification, confidence,
            bbox_alignment, palm_structure, occlusion_level,
            reasoning, model_name
        """
        raise NotImplementedError
