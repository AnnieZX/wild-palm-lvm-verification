"""Qwen2.5-VL verifier using Hugging Face Transformers."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from src.lvm.base_verifier import BaseVerifier
from src.lvm.prompt_template import build_verification_prompt
from src.lvm.response_schema import parse_json_response, validate_lvm_response


class QwenVerifier(BaseVerifier):
    """
    Verify palm annotations with Qwen2.5-VL via Hugging Face Transformers.

    model_name may be a Hugging Face repo id or a local checkpoint directory.
    """

    def __init__(
        self,
        model_name: str = "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
    ) -> None:
        super().__init__(model_name=model_name)
        self.device_map = device_map
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen2.5-VL model and processor."""
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Qwen2.5-VL.\n"
                "Install cluster requirements with:\n"
                "  pip install -r requirements_cluster.txt\n"
                f"Original error: {error}"
            ) from error

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_name}\n"
                "Download the model on the cluster first, or pass a valid Hugging Face repo id."
            )

        print(f"Loading Qwen model from: {self.model_name}")
        print(f"Device map: {self.device_map}")

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device_map,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception as error:
            raise RuntimeError(
                "Failed to load Qwen2.5-VL model.\n"
                "Possible causes:\n"
                "  - Model files missing or incomplete\n"
                "  - Insufficient GPU/RAM memory\n"
                "  - Missing or incompatible transformers installation\n"
                f"Original error: {error}"
            ) from error

        print("Qwen model loaded successfully.")

    @staticmethod
    def _print_processor_input_shapes(inputs: Any) -> None:
        """Print tensor shapes produced by the processor."""
        if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            print(f"  input_ids shape:      {tuple(inputs.input_ids.shape)}")

        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            print(f"  pixel_values shape:   {tuple(inputs.pixel_values.shape)}")
        else:
            print("  pixel_values:         not present")

        if hasattr(inputs, "keys"):
            for key in inputs.keys():
                if key in {"input_ids", "pixel_values"}:
                    continue
                value = inputs[key]
                if hasattr(value, "shape"):
                    print(f"  {key} shape: {tuple(value.shape)}")

    def generate_response_debugged(
        self,
        *,
        sample_id: str,
        image_path: Path | str,
        prompt_path: Path | str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Run one image + text inference with verbose pre-generation diagnostics.

        Exceptions propagate to the caller after debug context is printed.
        """
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image_path = Path(image_path).resolve()
        prompt_path = Path(prompt_path)

        print(f"\n=== Inference debug: {sample_id} ===")
        print(f"  sample_id:     {sample_id}")
        print(f"  image_path:    {image_path}")
        print(f"  image_exists:  {image_path.exists()}")
        print(f"  prompt_path:   {prompt_path}")
        print(f"  prompt_exists: {prompt_path.exists()}")
        print(f"  prompt_length: {len(prompt)}")

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as loaded_image:
            print(f"  image_size:    {loaded_image.size} (width, height)")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        print(f"  messages:      {messages}")

        print("  calling apply_chat_template() ...")
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"  chat_template length: {len(text)} characters")

        print("  calling process_vision_info() ...")
        image_inputs, video_inputs = process_vision_info(messages)

        print("  calling processor() ...")
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        self._print_processor_input_shapes(inputs)

        print("  moving inputs to model device ...")
        inputs = inputs.to(self.model.device)
        self._print_processor_input_shapes(inputs)

        print("  calling model.generate() ...")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"  generation complete for {sample_id}")
        return output_text[0]

    def _generate_response(self, image_path: Path, prompt: str) -> str:
        """Run one image + text inference and return raw model text."""
        from qwen_vl_utils import process_vision_info

        image_path = image_path.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def generate_batch_responses(
        self,
        items: list[tuple[Path | str, str]],
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Run batch inference for multiple (image_path, prompt) pairs."""
        if not items:
            return []

        from qwen_vl_utils import process_vision_info

        texts: list[str] = []
        image_input_batches: list[Any] = []
        video_input_batches: list[Any] = []

        for image_path, prompt in items:
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
                }
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            image_inputs, video_inputs = process_vision_info(messages)
            image_input_batches.append(image_inputs)
            video_input_batches.append(video_inputs)

        inputs = self.processor(
            text=texts,
            images=image_input_batches,
            videos=video_input_batches,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def verify_image(
        self,
        image_path: str,
        metadata: dict,
        prompt: str | None = None,
        use_legacy_validation: bool = True,
    ) -> dict:
        """
        Verify one palm overlay image with Qwen2.5-VL.

        Args:
            image_path: Path to the input image.
            metadata: Palm metadata dictionary.
            prompt: Optional custom prompt. Uses the default template when omitted.
            use_legacy_validation: When False, skip legacy schema validation and
                return only raw_response plus basic metadata fields.

        Returns a dictionary containing raw_response and, when parsing succeeds,
        validated schema fields.
        """
        image_path = Path(image_path)
        if prompt is None:
            prompt = build_verification_prompt(metadata)

        print(f"Running Qwen verification for: {image_path.name}")
        raw_response = self._generate_response(image_path, prompt)

        result: dict = {
            "image_name": metadata.get("image_name", image_path.name),
            "palm_id": metadata.get("palm_id", "unknown"),
            "raw_response": raw_response,
            "model_name": self.model_name,
        }

        if not use_legacy_validation:
            return result

        try:
            parsed = parse_json_response(raw_response)
            validated = validate_lvm_response(parsed)
            result.update(validated)
        except ValueError as error:
            result["parse_error"] = str(error)
            print(f"Warning: could not parse/validate model JSON: {error}")

        return result
