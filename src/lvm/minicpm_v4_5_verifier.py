"""MiniCPM-V-4.5 verifier (Hugging Face Transformers + trust_remote_code)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MiniCpmV45Verifier:
    """
    Load openbmb/MiniCPM-V-4_5 and run single-sample image+text generation.

    Follows the official Hugging Face Quick Start for MiniCPM-V-4.5:
      AutoModel + AutoTokenizer with trust_remote_code=True
      attn_implementation='sdpa' (flash_attention_2 optional)
      torch_dtype=torch.bfloat16
      msgs = [{'role': 'user', 'content': [image, question]}]
      model.chat(msgs=..., tokenizer=..., enable_thinking=False, ...)

    Generation parameter mapping (documented, not silent):
      Frozen protocol target: do_sample=False (deterministic).
      Official chat() API uses ``sampling`` (bool), not ``do_sample``.
      When sampling=True, chat() builds a config that includes do_sample and
      accepts kwargs overrides; we pass sampling=True + do_sample=False to
      achieve greedy decoding through the official chat path.
      (Official sampling=False instead uses num_beams=3 beam search.)
      enable_thinking=False keeps hybrid deep-thinking off.
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/MiniCPM-V-4_5",
        device_map: str = "auto",
        *,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = bool(enable_thinking)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if self.dtype in {"float16", "fp16"}:
            return torch.float16
        if self.dtype in {"auto", None, ""}:
            return "auto"
        raise ValueError(f"Unsupported dtype for MiniCPM-V-4.5: {self.dtype!r}")

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for MiniCPM-V-4.5.\n"
                "Use the isolated env /deac/csc/yangGrp/luoz23/envs/wild-palm-minicpm45 "
                "(transformers==4.51.0, torch, torchvision, accelerate).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download openbmb/MiniCPM-V-4_5 to the cluster path first."
            )

        torch_dtype = self._resolve_torch_dtype(torch)
        print(f"Loading MiniCPM-V-4.5 from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"dtype: {self.dtype} -> {torch_dtype}")
        print(f"attn_implementation: {self.attn_implementation}")
        print(f"trust_remote_code: {self.trust_remote_code}")
        print(f"enable_thinking: {self.enable_thinking}")

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch_dtype,
            "attn_implementation": self.attn_implementation,
        }
        # Prefer device_map when requested; fall back to .cuda() if auto fails
        # on single-GPU nodes (official card uses .eval().cuda()).
        if self.device_map and self.device_map != "none":
            load_kwargs["device_map"] = self.device_map

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            self.model = self.model.eval()
            if self.device_map in {"none", "", None}:
                self.model = self.model.cuda()
        except Exception as error:
            raise RuntimeError(
                "Failed to load MiniCPM-V-4.5.\n"
                "Possible causes: incomplete checkpoint, incompatible "
                "transformers (!=4.51.0), missing trust_remote_code files, "
                "or insufficient GPU memory.\n"
                f"Original error: {error}"
            ) from error

        print("MiniCPM-V-4.5 model loaded successfully.")

    def generate_response(
        self,
        *,
        image_path: Path | str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Run one image + frozen prompt inference via official model.chat().

        Semantic prompt text is unchanged; only MiniCPM message wrapping
        (PIL image + text in content list) is model-specific.
        """
        from PIL import Image

        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Official chat format: content is [PIL.Image, question_str].
        msgs = [{"role": "user", "content": [image, prompt]}]

        # Deterministic generation via official chat() API:
        #   sampling=True selects the branch that exposes do_sample in
        #   generation_config; do_sample=False overrides to greedy decode.
        #   enable_thinking=False disables hybrid deep-thinking mode.
        answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            sampling=True,
            do_sample=False,
            enable_thinking=self.enable_thinking,
            stream=False,
        )
        if isinstance(answer, str):
            return answer.strip()
        return str(answer).strip()
