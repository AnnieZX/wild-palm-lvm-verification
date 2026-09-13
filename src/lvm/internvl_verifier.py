"""InternVL3 multimodal verifier using the official Hugging Face remote-code API."""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ImageNet normalization constants from the InternVL3 model card Quick Start.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLVerifier:
    """
    Load InternVL3 instruction-tuned chat models and run single-sample generation.

    Follows OpenGVLab InternVL3 Hugging Face usage:
      AutoModel + AutoTokenizer with trust_remote_code=True,
      dynamic 448px tiling, and model.chat(...).
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B-Instruct",
        device_map: str = "auto",
        *,
        max_num_tiles: int = 12,
        use_flash_attn: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.max_num_tiles = max_num_tiles
        self.use_flash_attn = use_flash_attn
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load InternVL3 via AutoModel / AutoTokenizer (trust_remote_code)."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for InternVL3.\n"
                "Requires transformers>=4.37.2 (official InternVL3 card).\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download InternVL3-8B-Instruct to the cluster path first."
            )

        print(f"Loading InternVL3 from: {self.model_name}")
        print(f"Device map: {self.device_map}")
        print(f"use_flash_attn: {self.use_flash_attn}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False,
            )
            load_kwargs: dict[str, Any] = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_flash_attn": self.use_flash_attn,
            }
            # Single-GPU jobs: prefer explicit cuda placement like the model card.
            if self.device_map in ("auto", "cuda", "cuda:0"):
                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        **load_kwargs,
                    )
                    self.model = self.model.eval().cuda()
                except Exception as flash_error:
                    if not self.use_flash_attn:
                        raise
                    print(
                        "InternVL3 load with flash-attn failed; "
                        f"retrying without flash-attn. Error: {flash_error}"
                    )
                    load_kwargs["use_flash_attn"] = False
                    self.use_flash_attn = False
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        **load_kwargs,
                    )
                    self.model = self.model.eval().cuda()
            else:
                load_kwargs["device_map"] = self.device_map
                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        **load_kwargs,
                    ).eval()
                except Exception as flash_error:
                    if not self.use_flash_attn:
                        raise
                    print(
                        "InternVL3 load with flash-attn failed; "
                        f"retrying without flash-attn. Error: {flash_error}"
                    )
                    load_kwargs["use_flash_attn"] = False
                    self.use_flash_attn = False
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        **load_kwargs,
                    ).eval()
        except OSError as error:
            raise RuntimeError(
                "Failed to download or load InternVL3 weights/tokenizer.\n"
                "Check network access, disk space, and checkpoint completeness.\n"
                f"Original error: {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "Failed to load InternVL3 model.\n"
                "Possible causes:\n"
                "  - Checkpoint incomplete or wrong architecture\n"
                "  - Insufficient GPU memory\n"
                "  - Incompatible transformers / missing remote code deps\n"
                f"Original error: {error}"
            ) from error

        print("InternVL3 loaded successfully.")

    @staticmethod
    def _build_transform(input_size: int = 448):
        """Official InternVL image transform (ImageNet mean/std, bicubic resize)."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    @staticmethod
    def _find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        """Select the dynamic-tile aspect ratio closest to the image aspect."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @classmethod
    def _dynamic_preprocess(
        cls,
        image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> list:
        """Official InternVL dynamic tiling (448×448 patches + optional thumbnail)."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = cls._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def load_image(self, image_file: Path | str, input_size: int = 448):
        """Load and tile an image into InternVL pixel_values (official path)."""
        import torch
        from PIL import Image

        image = Image.open(image_file).convert("RGB")
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=self.max_num_tiles,
        )
        pixel_values = torch.stack([transform(tile) for tile in images])
        return pixel_values

    @staticmethod
    def build_internvl_question(prompt: str) -> str:
        """
        Wrap the frozen verification prompt with InternVL's required image token.

        Semantic instruction text is unchanged; only the official `<image>` marker
        is prepended as required by model.chat (see InternVL3 HF Quick Start).
        """
        if prompt.lstrip().startswith("<image>"):
            return prompt
        return f"<image>\n{prompt}"

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

        pixel_values = self.load_image(image_path).to(torch.bfloat16).cuda()
        question = self.build_internvl_question(prompt)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
            )

        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()
