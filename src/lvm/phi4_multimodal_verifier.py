"""Phi-4 Multimodal Instruct verifier (Hugging Face Transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Phi4MultimodalVerifier:
    """
    Load microsoft/Phi-4-multimodal-instruct and run single-sample image+text generation.

    Follows the official model-card loading path:
      AutoProcessor + AutoModelForCausalLM with trust_remote_code=True
      prompt: <|user|><|image_1|>{text}<|end|><|assistant|>
      processor(text=..., images=PIL.Image, return_tensors="pt")
      model.generate(..., do_sample=False)

    Vision/speech LoRA adapters are loaded by the remote modeling code when present
    in the checkpoint. Audio is unused for this verification task.
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/Phi-4-multimodal-instruct",
        device_map: str = "auto",
        *,
        dtype: str = "auto",
        attn_implementation: str = "eager",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.processor = None
        self.generation_config = None
        self._load_model()

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if self.dtype in {"float16", "fp16"}:
            return torch.float16
        return "auto"

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Phi-4 Multimodal Instruct.\n"
                "Use the isolated env /deac/csc/yangGrp/luoz23/envs/wild-palm-phi4 "
                "(transformers>=4.48.2, peft, torch).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download microsoft/Phi-4-multimodal-instruct first."
            )

        print(f"Loading Phi-4-Multimodal-Instruct from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype}")
        print(f"attn_implementation: {self.attn_implementation}")
        print(f"trust_remote_code: {self.trust_remote_code}")

        torch_dtype = self._resolve_torch_dtype(torch)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    _attn_implementation=self.attn_implementation,
                    **load_kwargs,
                )
            except Exception as flash_error:
                if self.attn_implementation != "eager":
                    print(
                        "Flash/SDPA attention load failed; retrying with eager.\n"
                        f"  {type(flash_error).__name__}: {flash_error}"
                    )
                    self.attn_implementation = "eager"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        _attn_implementation="eager",
                        **load_kwargs,
                    )
                else:
                    raise
            self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        except Exception as error:
            raise RuntimeError(
                "Failed to load Phi-4-Multimodal-Instruct.\n"
                "Possible causes: incomplete checkpoint, missing peft/remote code, "
                "incompatible transformers, or insufficient GPU memory.\n"
                f"Original error: {error}"
            ) from error

        self.model.eval()
        print("Phi-4-Multimodal-Instruct loaded successfully.")

    @staticmethod
    def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
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
        """Run one image + text inference and return generated text only."""
        import torch
        from PIL import Image

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Official vision chat format from the model card.
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        formatted = (
            f"{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}"
        )

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=formatted,
            images=image,
            return_tensors="pt",
        )

        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self._move_inputs_to_device(inputs, device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if self.generation_config is not None:
            gen_kwargs["generation_config"] = self.generation_config

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[:, input_len:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
