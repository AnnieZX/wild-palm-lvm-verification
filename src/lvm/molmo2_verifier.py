"""Molmo2-8B verifier (Allen Institute for AI / Hugging Face Transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Molmo2Verifier:
    """
    Load allenai/Molmo2-8B and run single-sample image+text generation.

    Follows the official Hugging Face model card Quick Start:
      AutoProcessor + AutoModelForImageTextToText
      trust_remote_code=True
      processor.apply_chat_template(..., tokenize=True, return_dict=True)
      model.generate under torch.inference_mode()
      decode only newly generated tokens via processor.tokenizer.decode

    Official install pin: transformers==4.57.1 plus torch, pillow, einops,
    torchvision, accelerate, decord2, molmo_utils. Use the isolated env
    /deac/csc/yangGrp/luoz23/envs/wild-palm-molmo2.

    Frozen prompt text is unchanged; only chat wrapping is model-specific.
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/Molmo2-8B",
        device_map: str = "auto",
        *,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
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
        raise ValueError(f"Unsupported dtype for Molmo2: {self.dtype!r}")

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Molmo2-8B.\n"
                "Use the isolated env /deac/csc/yangGrp/luoz23/envs/wild-palm-molmo2 "
                "(transformers==4.57.1, molmo_utils, torch, einops, accelerate).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download allenai/Molmo2-8B to the cluster path first "
                "(jobs/download_molmo2_8b.slurm)."
            )

        torch_dtype = self._resolve_torch_dtype(torch)
        print(f"Loading Molmo2-8B from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype} -> {torch_dtype}")
        print(f"trust_remote_code: {self.trust_remote_code}")

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": torch_dtype,
            "device_map": self.device_map,
        }

        try:
            # Official card passes dtype/device_map to the processor as well.
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                dtype=torch_dtype,
                device_map=self.device_map,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load Molmo2-8B.\n"
                "Possible causes: incomplete checkpoint, incompatible "
                "transformers (need ==4.57.1), missing molmo_utils / remote "
                "code, or insufficient GPU memory.\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("Molmo2-8B model loaded successfully.")

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

        Uses the official Molmo2 Transformers chat-template path for images
        (multi-image QA pattern with a single PIL image). Generation is
        deterministic (do_sample=False).
        """
        import torch
        from PIL import Image

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Official multi-image QA order: text then image content blocks.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = self._move_inputs_to_model(inputs, self.model)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = int(inputs["input_ids"].shape[-1])
        generated_tokens = generated_ids[0, input_len:]
        return self.processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
