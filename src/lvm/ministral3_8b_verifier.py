"""Ministral 3 Instruct 2512 verifier (BF16 checkpoints; 3B / 8B / 14B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Ministral3Verifier:
    """
    Load mistralai/Ministral-3-8B-Instruct-2512-BF16 and run single-sample
    image+text generation via Hugging Face Transformers (>= 5.0).

    HF-native API:
      AutoProcessor (PixtralProcessor) + Mistral3ForConditionalGeneration
      model.generate(..., do_sample=False)
      decode only newly generated tokens

    Prompt rendering: ``[INST][IMG]{prompt}[/INST]`` (BOS added by the
    processor). This is token-identical to Mistral's reference tokenizer
    (mistral-common, used by the model card's Transformers snippet and vLLM)
    for a single user turn with one image. The repo's HF Jinja chat template
    instead injects a ~520-token default "Le Chat" system prompt (persona,
    unfilled ``{today}`` date, tool-use and ask-to-clarify instructions) that
    is not part of the frozen verification prompt, so it is not used unless
    ``use_hf_default_system_prompt=True``.
    """

    EOS_TOKEN_ID = 2
    PAD_TOKEN_ID = 11

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/Ministral-3-8B-Instruct-2512-BF16",
        device_map: str = "auto",
        *,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        use_hf_default_system_prompt: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.use_hf_default_system_prompt = use_hf_default_system_prompt
        self.model = None
        self.processor = None
        self._torch_dtype = None
        self._load_model()

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if self.dtype in {"float16", "fp16"}:
            return torch.float16
        raise ValueError(f"Unsupported dtype for Ministral 3: {self.dtype!r}")

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Mistral3ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Ministral 3 8B.\n"
                "Requires transformers >= 5.0 (mistral3 + ministral3 text config). "
                "Use the isolated env /deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4.\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download mistralai/Ministral-3-8B-Instruct-2512-BF16 first "
                "(jobs/download_ministral3_8b.slurm)."
            )

        self._torch_dtype = self._resolve_torch_dtype(torch)
        print(f"Loading Ministral 3 Instruct from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype} -> {self._torch_dtype}")
        print(f"attn_implementation: {self.attn_implementation}")
        print(f"use_hf_default_system_prompt: {self.use_hf_default_system_prompt}")

        try:
            # fix_mistral_regex: Transformers flags the shipped tokenizer.json
            # pre-tokenizer regex as incorrect for Mistral tokenizers.
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                fix_mistral_regex=True,
            )
            load_kwargs: dict[str, Any] = {
                "dtype": self._torch_dtype,
                "device_map": self.device_map,
            }
            if self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load Ministral 3 Instruct from {self.model_name}.\n"
                "Possible causes: incomplete checkpoint, transformers < 5.0, or "
                "insufficient GPU memory.\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("Ministral 3 Instruct loaded successfully.")
        print(f"Generation contract: {self.generation_kwargs(512)}")

    def generation_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        """Explicit greedy decoding; overrides checkpoint generation_config."""
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "num_beams": 1,
            "eos_token_id": self.EOS_TOKEN_ID,
            "pad_token_id": self.PAD_TOKEN_ID,
        }

    def build_inputs(self, image: Any, prompt: str) -> Any:
        """Tokenize one image + frozen prompt (image first, then text)."""
        if self.use_hf_default_system_prompt:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        text = f"[INST]{self.processor.image_token}{prompt}[/INST]"
        return self.processor(text=text, images=[image], return_tensors="pt")

    def render_prompt(self, image: Any, prompt: str) -> str:
        """Decode the exact token sequence the model sees (for audit)."""
        inputs = self.build_inputs(image, prompt)
        return self.processor.tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=False,
        )

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
        n_image_tokens = int((inputs["input_ids"] == self.processor.image_token_id).sum())
        if n_image_tokens == 0:
            raise RuntimeError("No image tokens in rendered input.")

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
