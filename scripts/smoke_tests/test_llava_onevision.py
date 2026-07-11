#!/usr/bin/env python3
"""
Standalone smoke test for LLaVA-OneVision on the DEAC cluster.

Validates that the model can be loaded and run one image-description inference
before adapter integration into the verification framework.

Example:
    python scripts/smoke_tests/test_llava_onevision.py \\
        --image data/samples/images/100_0003_0001_1.png
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

DEFAULT_MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_PROMPT = "Briefly describe this image in one or two sentences."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test LLaVA-OneVision load + single-image inference.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a local PNG/JPEG image for inference",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local checkpoint directory (overrides --model-id when set)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    return parser.parse_args()


def fail(message: str, error: Exception | None = None) -> None:
    """Print a clear failure message and exit."""
    print()
    print("Smoke test FAILED")
    print(message)
    if error is not None:
        print(f"Details: {error}")
        traceback.print_exc()
    sys.exit(1)


def resolve_model_source(args: argparse.Namespace) -> str:
    if args.model_path is not None:
        if not args.model_path.exists():
            fail(f"Local model path not found: {args.model_path}")
        return str(args.model_path.resolve())
    return args.model_id


def check_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        fail(
            "PyTorch is not installed. Install cluster requirements:\n"
            "  pip install -r requirements_cluster.txt",
            error,
        )

    if not torch.cuda.is_available():
        fail(
            "CUDA is not available on this node.\n"
            "Run this smoke test on a GPU node (e.g. DEAC L40S via srun/sbatch)."
        )

    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device: {device_name}")


def import_transformers_classes():
    try:
        from transformers import (
            LlavaOnevisionForConditionalGeneration,
            LlavaOnevisionProcessor,
        )
    except ImportError as error:
        fail(
            "Could not import LLaVA-OneVision classes from transformers.\n"
            "This model requires transformers >= 4.45 with LLaVA-OneVision support.\n"
            "Upgrade on the cluster, for example:\n"
            "  pip install -U 'transformers>=4.45.0'",
            error,
        )
    return LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor


def load_image(image_path: Path):
    try:
        from PIL import Image
    except ImportError as error:
        fail("Pillow is required to load images.", error)

    if not image_path.is_file():
        fail(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as error:
        fail(f"Failed to open image: {image_path}", error)

    return image


def build_conversation(prompt_text: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def move_inputs_to_model(inputs, model, dtype):
    """Move processor outputs onto the model device with the inference dtype."""
    if hasattr(inputs, "to"):
        return inputs.to(model.device, dtype=dtype)

    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if value.is_floating_point():
                moved[key] = value.to(model.device, dtype=dtype)
            else:
                moved[key] = value.to(model.device)
        else:
            moved[key] = value
    return moved


def main() -> None:
    args = parse_args()
    model_source = resolve_model_source(args)

    print("LLaVA-OneVision smoke test")
    print(f"  Model: {model_source}")
    print(f"  Image: {args.image.resolve()}")
    print()

    check_cuda()

    import torch

    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor = import_transformers_classes()
    image = load_image(args.image)

    print("Loading processor...")
    try:
        processor = LlavaOnevisionProcessor.from_pretrained(model_source)
    except OSError as error:
        fail(
            "Failed to download or load the processor.\n"
            "Check network access, Hugging Face credentials, and disk space.",
            error,
        )
    except Exception as error:
        fail("Unexpected error while loading LlavaOnevisionProcessor.", error)

    print("Loading model...")
    try:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except OSError as error:
        fail(
            "Failed to download or load model weights.\n"
            "Check network access, Hugging Face credentials, and GPU memory.",
            error,
        )
    except RuntimeError as error:
        fail(
            "Failed to load model onto GPU.\n"
            "The checkpoint may be incompatible or GPU memory may be insufficient.",
            error,
        )
    except Exception as error:
        fail("Unexpected error while loading LlavaOnevisionForConditionalGeneration.", error)

    model.eval()
    conversation = build_conversation(DEFAULT_PROMPT)

    print("Running inference...")
    started_at = time.perf_counter()
    try:
        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = move_inputs_to_model(inputs, model, torch.bfloat16)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0, input_len:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    except (ValueError, TypeError) as error:
        fail(
            "Processor/model input mismatch while preparing inputs.\n"
            "The checkpoint may not match LlavaOnevisionProcessor.",
            error,
        )
    except RuntimeError as error:
        fail("Generation failed during model.generate().", error)
    except Exception as error:
        fail("Unexpected error during inference.", error)

    elapsed = time.perf_counter() - started_at

    print()
    print(f"Inference time: {elapsed:.2f}s")
    print()
    print("Generated response:")
    print(response)
    print()
    print("Smoke test PASSED")


if __name__ == "__main__":
    main()
