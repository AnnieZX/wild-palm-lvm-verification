"""Qwen3-VL-8B-Instruct verifier (Hugging Face Transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Qwen3VlVerifier:
    """
    Load Qwen/Qwen3-VL-8B-Instruct and run single-sample image+text generation.

    Follows the official Hugging Face Quick Start for Qwen3-VL:
      Qwen3VLForConditionalGeneration + AutoProcessor
      processor.apply_chat_template(..., tokenize=True, return_dict=True)
      model.generate(..., do_sample=False)

    This is intentionally separate from Qwen2.5-VL (different model class and
    chat-template tokenization path). Frozen prompt text is unchanged.
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/Qwen3-VL-8B-Instruct",
        device_map: str = "auto",
        *,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.model = None
        self.processor = None
        self._load_model()

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if self.dtype in {"float16", "fp16"}:
            return torch.float16
        if self.dtype in {"auto", None, ""}:
            return "auto"
        raise ValueError(f"Unsupported dtype for Qwen3-VL: {self.dtype!r}")

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Qwen3-VL.\n"
                "Use the isolated env /deac/csc/yangGrp/luoz23/envs/wild-palm-qwen3vl "
                "(transformers>=4.57.0, qwen-vl-utils, torch).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download Qwen/Qwen3-VL-8B-Instruct to the cluster path first."
            )

        torch_dtype = self._resolve_torch_dtype(torch)
        print(f"Loading Qwen3-VL from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype} -> {torch_dtype}")
        print(f"attn_implementation: {self.attn_implementation}")

        load_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": self.device_map,
        }
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load Qwen3-VL-8B-Instruct.\n"
                "Possible causes: incomplete checkpoint, incompatible "
                "transformers (<4.57), or insufficient GPU memory.\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("Qwen3-VL model loaded successfully.")

    @staticmethod
    def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                return inputs
        if hasattr(inputs, "to"):
            return inputs.to(device)
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def generate_response(
        self,
        *,
        image_path: Path | str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Run one image + frozen prompt inference.

        Uses the official Qwen3-VL Transformers chat-template path
        (tokenize=True, return_dict=True). Generation is deterministic.
        """
        import torch
        from PIL import Image

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Semantic prompt text is unchanged; only chat wrapping is model-specific.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_model(inputs, self.model)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = int(inputs["input_ids"].shape[-1])
        trimmed = generated_ids[0][input_len:]
        return self.processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
