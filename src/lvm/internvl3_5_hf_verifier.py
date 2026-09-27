"""InternVL3.5-8B-HF verifier (Hugging Face Transformers native API)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class InternVL35HfVerifier:
    """
    Load OpenGVLab/InternVL3_5-8B-HF and run single-sample image+text generation.

    Follows the official Transformers InternVL HF-format API
    (docs: transformers model_doc/internvl):
      AutoProcessor + AutoModelForImageTextToText (InternVLForConditionalGeneration)
      processor.apply_chat_template(..., tokenize=True, return_dict=True)
      model.generate(..., do_sample=False)

    This is intentionally separate from InternVL3-8B-Instruct (remote-code
    AutoModel + model.chat). Frozen prompt text is unchanged.
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/InternVL3_5-8B-HF",
        device_map: str = "auto",
        *,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
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
        raise ValueError(f"Unsupported dtype for InternVL3.5-HF: {self.dtype!r}")

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for InternVL3.5-HF.\n"
                "Requires transformers>=4.52.1 with native InternVL support "
                "(AutoProcessor + AutoModelForImageTextToText).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download OpenGVLab/InternVL3_5-8B-HF to the cluster path first."
            )

        torch_dtype = self._resolve_torch_dtype(torch)
        print(f"Loading InternVL3.5-HF from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype} -> {torch_dtype}")
        print(f"trust_remote_code: {self.trust_remote_code}")
        print(f"attn_implementation: {self.attn_implementation}")

        load_kwargs: dict[str, Any] = {
            # Transformers >=4.55 prefers `dtype=` (torch_dtype still works with a warning).
            "dtype": torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load InternVL3.5-8B-HF.\n"
                "Possible causes: incomplete checkpoint, incompatible "
                "transformers (<4.52.1), or insufficient GPU memory.\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("InternVL3.5-HF model loaded successfully.")

    @staticmethod
    def _move_inputs_to_model(inputs: Any, model: Any, torch_dtype: Any) -> Any:
        """Place tensors on the model device; cast floating inputs to model dtype."""
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                return inputs

        if hasattr(inputs, "to"):
            # BatchEncoding / Tensor-like: move first, then cast float tensors.
            moved = inputs.to(device)
            try:
                for key in list(moved.keys()):
                    value = moved[key]
                    if hasattr(value, "is_floating_point") and value.is_floating_point():
                        moved[key] = value.to(dtype=torch_dtype)
            except Exception:
                pass
            return moved

        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(device)
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    value = value.to(dtype=torch_dtype)
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

        Uses the official HF InternVL chat-template path
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
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        torch_dtype = self._resolve_torch_dtype(torch)
        inputs = self._move_inputs_to_model(inputs, self.model, torch_dtype)

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
