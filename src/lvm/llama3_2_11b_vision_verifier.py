"""Llama 3.2 11B Vision Instruct verifier (meta-llama/Llama-3.2-11B-Vision-Instruct)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_llama3_2_vision_inputs(processor: Any, image: Any, prompt: str) -> Any:
    """
    Tokenize one image + frozen prompt with the official Mllama chat template.

    Official model-card path: image content block, then text; the chat
    template emits ``<|begin_of_text|>`` itself, so the processor must be
    called with ``add_special_tokens=False`` (otherwise BOS is duplicated).
    With an image and no system message, the official template emits no
    system header (no "Cutting Knowledge Date" / "Today Date" block).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(
        images=image,
        text=text,
        add_special_tokens=False,
        return_tensors="pt",
    )


class Llama3_2VisionVerifier:
    """
    Load meta-llama/Llama-3.2-11B-Vision-Instruct and run single-sample
    image+text generation via Hugging Face Transformers.

    HF-native API:
      MllamaProcessor + MllamaForConditionalGeneration
      model.generate(..., do_sample=False)
      decode only newly generated tokens
    """

    # <|end_of_text|>, <|eom_id|>, <|eot_id|> (checkpoint generation_config).
    EOS_TOKEN_IDS = (128001, 128008, 128009)
    # <|finetune_right_pad_id|>
    PAD_TOKEN_ID = 128004

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/Llama-3.2-11B-Vision-Instruct",
        device_map: str = "auto",
        *,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.model = None
        self.processor = None
        self._torch_dtype = None
        self._load_model()

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if self.dtype in {"float16", "fp16"}:
            return torch.float16
        raise ValueError(f"Unsupported dtype for Llama 3.2 Vision: {self.dtype!r}")

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import MllamaForConditionalGeneration, MllamaProcessor
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Llama 3.2 11B Vision.\n"
                "Requires transformers >= 4.45 (mllama).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download meta-llama/Llama-3.2-11B-Vision-Instruct first "
                "(gated; requires approved Meta license on the HF account)."
            )

        self._torch_dtype = self._resolve_torch_dtype(torch)
        print(f"Loading Llama 3.2 11B Vision Instruct from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype} -> {self._torch_dtype}")
        print(f"attn_implementation: {self.attn_implementation}")

        try:
            self.processor = MllamaProcessor.from_pretrained(self.model_name)
            load_kwargs: dict[str, Any] = {
                "dtype": self._torch_dtype,
                "device_map": self.device_map,
            }
            if self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load Llama 3.2 11B Vision Instruct.\n"
                "Possible causes: incomplete checkpoint, transformers without mllama, or "
                "insufficient GPU memory (~21.3GB BF16 weights).\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("Llama 3.2 11B Vision Instruct loaded successfully.")
        print(f"Generation contract: {self.generation_kwargs(512)}")

    def generation_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        """Explicit greedy decoding; overrides checkpoint sampling defaults."""
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "num_beams": 1,
            "eos_token_id": list(self.EOS_TOKEN_IDS),
            "pad_token_id": self.PAD_TOKEN_ID,
        }

    def build_inputs(self, image: Any, prompt: str) -> Any:
        return build_llama3_2_vision_inputs(self.processor, image, prompt)

    def generate_response(
        self,
        *,
        image_path: Path | str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Run one image + frozen prompt inference (greedy)."""
        import torch
        from PIL import Image

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        inputs = self.build_inputs(image, prompt)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image|>")
        n_image_tokens = int((inputs["input_ids"] == image_token_id).sum())
        if n_image_tokens != 1:
            raise RuntimeError(f"Expected exactly 1 image token, got {n_image_tokens}.")

        device = self.model.device
        inputs = {
            key: (
                value.to(device, dtype=self._torch_dtype)
                if key == "pixel_values"
                else value.to(device)
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs(max_new_tokens),
            )

        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0, input_len:]
        if int(generated_ids.shape[-1]) >= max_new_tokens:
            print(f"WARNING: generation hit max_new_tokens={max_new_tokens} (possible truncation)")
        return self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
