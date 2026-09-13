#!/usr/bin/env python3
"""CPU-only prompt/input parity check: Qwen vs InternVL semantic instruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from src.lvm.internvl_verifier import InternVLVerifier
from src.paths import PROJECT_ROOT
from src.verification.jobs import load_verification_jobs, validate_jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-index",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/diagnostics/model_qualification/stage0_A1_10/prompt_index.csv",
    )
    parser.add_argument("--sample-id", type=str, default=None)
    args = parser.parse_args()

    jobs = load_verification_jobs(prompt_index=args.prompt_index)
    validate_jobs(jobs)
    job = jobs[0]
    if args.sample_id:
        matching = [j for j in jobs if j.sample_id == args.sample_id]
        if not matching:
            print(f"sample_id not found: {args.sample_id}", file=sys.stderr)
            return 1
        job = matching[0]

    prompt = job.prompt_path.read_text(encoding="utf-8")
    with Image.open(job.image_path) as img:
        size = img.size
        mode = img.mode

    internvl_question = InternVLVerifier.build_internvl_question(prompt)

    # Qwen path: user text content is the raw prompt file (no rewrite).
    qwen_semantic = prompt
    # InternVL: official chat requires '<image>\n' prefix; body must match Qwen.
    internvl_body = internvl_question
    if internvl_body.startswith("<image>\n"):
        internvl_semantic = internvl_body[len("<image>\n") :]
    else:
        internvl_semantic = internvl_body

    print("=== Prompt / input parity (CPU) ===")
    print(f"sample_id:           {job.sample_id}")
    print(f"image_path:          {job.image_path}")
    print(f"image_exists:        {job.image_path.exists()}")
    print(f"image_size:          {size} mode={mode}")
    print(f"prompt_path:         {job.prompt_path}")
    print(f"prompt_exists:       {job.prompt_path.exists()}")
    print(f"prompt_chars:        {len(prompt)}")
    print(f"qwen_user_text_sha:  {hash(qwen_semantic)}")
    print(f"internvl_body_sha:   {hash(internvl_semantic)}")
    print(f"semantic_equal:      {qwen_semantic == internvl_semantic}")
    print()
    print("Qwen chat construction:")
    print("  role=user; content=[image, text=prompt_file_exact]")
    print("  system prompt: none (verification adapter)")
    print()
    print("InternVL chat construction:")
    print("  model.chat(tokenizer, pixel_values, question, generation_config)")
    print("  question = '<image>\\n' + prompt_file_exact")
    print("  system prompt: whatever InternVL remote-code conversation template inserts")
    print("  model-specific token: required '<image>' marker (not a verification instruction rewrite)")
    print("  do_sample=False; max_new_tokens=512; dtype=bfloat16; dynamic tiles<=12")
    print()
    if qwen_semantic != internvl_semantic:
        print("FAIL: semantic instruction mismatch after stripping InternVL image marker")
        return 1
    print("PASS: semantic verification instruction identical for Qwen and InternVL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
