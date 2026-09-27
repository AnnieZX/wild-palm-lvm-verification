"""Gemma 4 12B Unified verifier (google/gemma-4-12B-it via Hugging Face Transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Gemma4Verifier:
    """
    Load google/gemma-4-12B-it (Gemma 4 12B Unified IT) and run single-sample
    image+text generation.

    Official HF card API (Transformers >= 5.10.1):
      AutoProcessor + AutoModelForMultimodalLM
      processor.apply_chat_template(..., enable_thinking=False)
      model.generate under torch.inference_mode()
      decode only newly generated tokens

    Frozen prompt text is unchanged; only chat wrapping is model-specific.
    Thinking mode is explicitly disabled to keep decoding deterministic under
    the frozen verification protocol (do_sample=False).
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/gemma-4-12B-it",
        device_map: str = "auto",
        *,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        enable_thinking: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.enable_thinking = enable_thinking
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
        raise ValueError(f"Unsupported dtype for Gemma 4: {self.dtype!r}")

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Gemma 4 12B Unified.\n"
                "Requires transformers >= 5.10.1 with AutoModelForMultimodalLM "
                "(gemma4_unified). Use the isolated env "
                "/deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4.\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download google/gemma-4-12B-it to the cluster path first "
                "(jobs/download_gemma4_12b_it.slurm)."
            )

        torch_dtype = self._resolve_torch_dtype(torch)
        print(f"Loading Gemma 4 12B Unified from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype} -> {torch_dtype}")
        print(f"attn_implementation: {self.attn_implementation}")
        print(f"enable_thinking: {self.enable_thinking}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            load_kwargs: dict[str, Any] = {
                "dtype": torch_dtype,
                "device_map": self.device_map,
            }
            if self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load Gemma 4 12B Unified.\n"
                "Possible causes: incomplete checkpoint, transformers < 5.10.1 "
                "(need gemma4_unified / AutoModelForMultimodalLM), or "
                "insufficient GPU memory (~27GB BF16 weights).\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("Gemma 4 12B Unified loaded successfully.")

    @staticmethod
    def _move_inputs_to_model(inputs: Any, model: Any, dtype: Any) -> Any:
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                return inputs

        if hasattr(inputs, "to"):
            try:
                return inputs.to(device, dtype=dtype)
            except TypeError:
                return inputs.to(device)

        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.is_floating_point():
                    moved[key] = value.to(device, dtype=dtype)
                else:
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

        Uses the official Gemma 4 Transformers image chat path (image before
        text). Generation is deterministic (do_sample=False); thinking mode
        follows self.enable_thinking (default False for frozen protocol).
        """
        import torch
        from PIL import Image

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Official card order for images: image content block, then text.
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
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
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
        return self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
