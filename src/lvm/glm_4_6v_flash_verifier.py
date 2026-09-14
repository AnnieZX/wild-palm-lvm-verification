"""GLM-4.6V-Flash multimodal verifier (Hugging Face Transformers)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# Official GLM thinking delimiters used when Hybrid Reasoning is enabled.
_THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
_ANSWER_BLOCK_RE = re.compile(
    r"<answer>(.*?)</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_final_answer_text(raw_text: str) -> str:
    """
    Extract the final answer span from a GLM generation.

    Uses documented structure when present:
      - strip <think>...</think> blocks
      - if <answer>...</answer> exists, use its contents
      - otherwise use remaining text after thinking blocks

    Does not repair or rewrite JSON content.
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    answer_match = _ANSWER_BLOCK_RE.search(text)
    if answer_match:
        return answer_match.group(1).strip()

    without_think = _THINK_BLOCK_RE.sub("", text).strip()
    return without_think if without_think else text


class Glm46vFlashVerifier:
    """
    Load zai-org/GLM-4.6V-Flash and run single-sample image+text generation.

    Follows the official Hugging Face Quick Start:
      AutoProcessor + Glm4vForConditionalGeneration
      processor.apply_chat_template(..., tokenize=True, return_dict=True)
      model.generate(...)

    Hybrid Reasoning: prefer non-thinking mode via the officially documented
    chat-template switch ``enable_thinking=False`` (same switch used by the
    GLM-V README for vLLM/SGLang as chat_template_kwargs).
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash",
        device_map: str = "auto",
        *,
        enable_thinking: bool = False,
        dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.enable_thinking = bool(enable_thinking)
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Glm4vForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for GLM-4.6V-Flash.\n"
                "Requires transformers with Glm4vForConditionalGeneration "
                "(cluster has 4.57.x; official card also cites newer builds).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download zai-org/GLM-4.6V-Flash to the cluster path first."
            )

        print(f"Loading GLM-4.6V-Flash from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"enable_thinking: {self.enable_thinking}")
        print(f"dtype: {self.dtype}")

        torch_dtype: Any = "auto"
        if self.dtype in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16
        elif self.dtype in {"float16", "fp16"}:
            torch_dtype = torch.float16

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load GLM-4.6V-Flash.\n"
                "Possible causes: incomplete checkpoint, incompatible "
                "transformers, or insufficient GPU memory.\n"
                f"Original error: {error}"
            ) from error

        # Native Transformers 5.x path only: do not rewrite rope_parameters /
        # rope_scaling. Requires isolated env with transformers >= 5.0.0rc0.
        text_cfg = getattr(self.model.config, "text_config", None)
        rope_params = getattr(text_cfg, "rope_parameters", None) if text_cfg else None
        print(
            f"Native RoPE path: rope_parameters present={rope_params is not None}; "
            f"transformers patch bypassed"
        )

        self.model.eval()
        print("GLM-4.6V-Flash loaded successfully.")

    @staticmethod
    def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(model.device)
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
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
        """
        Run one image + text inference and return the full raw generation.

        Callers that need the parseable final answer should run
        ``extract_final_answer_text`` on the returned string.
        """
        import torch
        from PIL import Image

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Semantic prompt is unchanged; only chat-template wrapping is model-specific.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Official HF Quick Start uses apply_chat_template(...).
        # enable_thinking=False is the documented Hybrid Reasoning off switch
        # (GLM-V README / vLLM chat_template_kwargs).
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": self.enable_thinking,
        }
        try:
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Older processor builds may not accept enable_thinking kwarg.
            template_kwargs.pop("enable_thinking", None)
            print(
                "WARNING: apply_chat_template rejected enable_thinking; "
                "retrying without it. Thinking may remain enabled.",
                flush=True,
            )
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)

        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)
        elif isinstance(inputs, dict):
            inputs.pop("token_type_ids", None)

        inputs = self._move_inputs_to_model(inputs, self.model)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = int(inputs["input_ids"].shape[-1])
        trimmed = generated_ids[0][input_len:]
        # Keep special tokens so thinking delimiters remain recoverable.
        return self.processor.decode(trimmed, skip_special_tokens=False).strip()
