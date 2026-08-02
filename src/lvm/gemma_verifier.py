"""Gemma 3 multimodal verifier using Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class GemmaVerifier:
    """
    Load Gemma 3 instruction-tuned multimodal models and run single-sample generation.

    Uses Gemma3ForConditionalGeneration and AutoProcessor per the official HF API.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Gemma 3 processor and model."""
        try:
            import torch
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Gemma 3.\n"
                "Requires transformers >= 4.50 with Gemma3 support.\n"
                "Upgrade on the cluster, for example:\n"
                "  pip install -U 'transformers>=4.50.0'\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download Gemma 3 to the cluster path, or pass a valid "
                "Hugging Face repo id / local checkpoint."
            )

        print(f"Loading Gemma 3 from: {self.model_name}")
        print(f"Device map: {self.device_map}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            )
        except OSError as error:
            raise RuntimeError(
                "Failed to download or load Gemma 3 weights/processor.\n"
                "Check Hugging Face credentials (Gemma license), network access, "
                "and disk space.\n"
                f"Original error: {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "Failed to load Gemma 3 model.\n"
                "Possible causes:\n"
                "  - Checkpoint incomplete or wrong architecture\n"
                "  - Insufficient GPU memory\n"
                "  - Incompatible transformers version\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("Gemma 3 loaded successfully.")

    @staticmethod
    def _move_inputs_to_model(inputs: Any, model: Any, dtype: Any) -> Any:
        """Move processor outputs onto the model device with the inference dtype."""
        if hasattr(inputs, "to"):
            try:
                return inputs.to(model.device, dtype=dtype)
            except TypeError:
                return inputs.to(model.device)

        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.is_floating_point():
                    moved[key] = value.to(model.device, dtype=dtype)
                else:
                    moved[key] = value.to(model.device)
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
        """Run one image + text inference and return generated text only."""
        import torch

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_model(inputs, self.model, torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0, input_len:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()
