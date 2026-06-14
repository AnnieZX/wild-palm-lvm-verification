"""Qwen2.5-VL verifier using Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path

from src.lvm.base_verifier import BaseVerifier
from src.lvm.prompt_template import build_verification_prompt

# Set to True on the cluster when real model loading is implemented.
ENABLE_QWEN_INFERENCE = False


class QwenVerifier(BaseVerifier):
    """
    Verify palm annotations with Qwen2.5-VL via Hugging Face Transformers.

    This skeleton keeps the BaseVerifier interface ready for cluster deployment.
    Model loading and inference are not enabled in local development yet.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map: str = "auto",
    ) -> None:
        super().__init__(model_name=model_name)
        self.device_map = device_map
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen2.5-VL model and processor on the cluster."""
        if not ENABLE_QWEN_INFERENCE:
            raise NotImplementedError(
                "Qwen model loading is deferred to cluster deployment."
            )

        # Future cluster implementation will go here:
        # - import torch, transformers, qwen_vl_utils
        # - load Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
        # - load AutoProcessor.from_pretrained(...)
        raise NotImplementedError(
            "Qwen model loading is deferred to cluster deployment."
        )

    def verify_image(self, image_path: str, metadata: dict) -> dict:
        """
        Verify one palm overlay image with Qwen2.5-VL.

        Returns a dictionary containing raw_response and validated schema fields
        once cluster inference is enabled.
        """
        image_path = Path(image_path)
        prompt = build_verification_prompt(metadata)

        if self.model is None or self.processor is None:
            raise NotImplementedError(
                "Qwen model loading is deferred to cluster deployment."
            )

        # Future cluster implementation will go here:
        # - run image + prompt through the model
        # - parse JSON with parse_json_response()
        # - validate with validate_lvm_response()
        raise NotImplementedError(
            "Qwen inference is deferred to cluster deployment."
        )
