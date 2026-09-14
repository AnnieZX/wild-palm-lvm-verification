#!/usr/bin/env python3
"""
Prepare A5_prime (A5') inputs by extracting the exact A4 right-panel pixels.

A5' := A4[:, width // 2 :]

Does NOT recrop, resize, pad, or call build_a5_crop_only_image.
Does NOT modify existing A1–A5 artifacts.

Writes:
  outputs/verification_ablation_1000/A5_prime_a4_crop_panel/
    images/
    prompts/
    prompt_index.csv
    prompt_index_smoke20.csv
    PIXEL_IDENTITY_REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts.ablation_verification_prompts import (  # noqa: E402
    _IMAGE_A5_CROP_ONLY,
    _base_prompt,
    _format_metadata_section,
    _metadata_lines,
)
from src.prompts.ablation_verification_prompt_builder import (  # noqa: E402
    load_sample_metadata,
)
from src.prompts.verification_prompt import prompt_filename_for_sample  # noqa: E402

CONDITION_NAME = "A5_prime_a4_crop_panel"
DEFAULT_A4_DIR = (
    PROJECT_ROOT / "outputs/verification_ablation_1000/A4_overlay_crop_confidence"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "outputs/verification_ablation_1000" / CONDITION_NAME
)
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs/verification_dataset"

# Same 20 IDs as qwen_label_robustness_20 / InternVL label-semantics probe.
SMOKE20_SAMPLE_IDS = [
    "sample_000071",
    "sample_000105",
    "sample_000163",
    "sample_000165",
    "sample_000174",
    "sample_000189",
    "sample_000190",
    "sample_000208",
    "sample_000210",
    "sample_000216",
    "sample_000019",
    "sample_000025",
    "sample_000036",
    "sample_000074",
    "sample_000077",
    "sample_000088",
    "sample_000093",
    "sample_000129",
    "sample_000130",
    "sample_000140",
]


def build_a5_prime_prompt(metadata: dict[str, Any], sample_id: str) -> str:
    """A5-style crop-only wording; decision semantics identical to A4/A5."""
    metadata_section = _format_metadata_section(
        _metadata_lines(metadata, [("confidence", "YOLO confidence")])
    )
    return _base_prompt(
        sample_id=sample_id,
        condition=CONDITION_NAME,
        image_description=_IMAGE_A5_CROP_ONLY,
        metadata_section=metadata_section,
        metadata_instruction=(
            "Use YOLO confidence only as auxiliary context.\n"
            "Do not use bounding-box geometry beyond what is visible in the image."
        ),
    )


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def extract_right_panel(a4: np.ndarray) -> np.ndarray:
    """Exact A4 right half: A4[:, width // 2 :]. No copy-transform beyond slice."""
    width = a4.shape[1]
    return a4[:, width // 2 :]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare A5_prime from A4 right panels.")
    p.add_argument("--a4-dir", type=Path, default=DEFAULT_A4_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument(
        "--sample-count",
        type=int,
        default=1000,
        help="Number of samples (default 1000; uses A4 prompt_index order).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    a4_dir = args.a4_dir.resolve()
    out_dir = args.output_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()

    a4_index_path = a4_dir / "prompt_index.csv"
    if not a4_index_path.exists():
        print(f"ERROR: missing A4 prompt index: {a4_index_path}", file=sys.stderr)
        return 1

    a4_index = pd.read_csv(a4_index_path).head(args.sample_count)
    if len(a4_index) != args.sample_count:
        print(
            f"ERROR: expected {args.sample_count} A4 rows, found {len(a4_index)}",
            file=sys.stderr,
        )
        return 1

    images_dir = out_dir / "images"
    prompts_dir = out_dir / "prompts"
    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    expected_shape: tuple[int, ...] | None = None

    print("A5_prime preparation")
    print(f"  A4 dir:    {a4_dir}")
    print(f"  Output:    {out_dir}")
    print(f"  N:         {len(a4_index)}")
    print()

    for _, rec in a4_index.iterrows():
        sample_id = str(rec["sample_id"])
        a4_image_rel = str(rec["image_path"])
        a4_path = a4_dir / a4_image_rel
        if not a4_path.exists():
            failures.append(
                {"sample_id": sample_id, "reason": f"missing A4 image: {a4_path}"}
            )
            continue

        a4 = cv2.imread(str(a4_path), cv2.IMREAD_COLOR)
        if a4 is None:
            failures.append(
                {"sample_id": sample_id, "reason": f"failed to read A4: {a4_path}"}
            )
            continue

        right = extract_right_panel(a4)
        if expected_shape is None:
            expected_shape = tuple(right.shape)

        out_image_rel = f"images/{sample_id}.png"
        out_image_path = out_dir / out_image_rel
        # Write then re-load to validate on-disk decode identity.
        if not cv2.imwrite(str(out_image_path), right):
            failures.append(
                {"sample_id": sample_id, "reason": f"imwrite failed: {out_image_path}"}
            )
            continue

        loaded = cv2.imread(str(out_image_path), cv2.IMREAD_COLOR)
        if loaded is None:
            failures.append(
                {
                    "sample_id": sample_id,
                    "reason": f"failed to re-read A5_prime: {out_image_path}",
                }
            )
            continue

        max_abs = int(np.abs(loaded.astype(np.int16) - right.astype(np.int16)).max())
        n_diff = int(np.sum(np.any(loaded != right, axis=2)))
        shape_ok = tuple(loaded.shape) == tuple(right.shape)
        identity_ok = shape_ok and max_abs == 0 and n_diff == 0

        sample_report = {
            "sample_id": sample_id,
            "a4_shape": list(a4.shape),
            "a5_prime_shape": list(loaded.shape),
            "expected_right_shape": list(right.shape),
            "max_abs_diff": max_abs,
            "differing_pixels": n_diff,
            "sha256_right_panel": sha256_array(right),
            "sha256_a5_prime_decoded": sha256_array(loaded),
            "identity_ok": identity_ok,
        }
        per_sample.append(sample_report)

        if not identity_ok:
            failures.append(
                {
                    "sample_id": sample_id,
                    "reason": "pixel identity failed",
                    "max_abs_diff": max_abs,
                    "differing_pixels": n_diff,
                    "shape": list(loaded.shape),
                    "expected_shape": list(right.shape),
                }
            )
            continue

        # Prompt + metadata (confidence only; same as A4/A5).
        meta_path = dataset_dir / "metadata" / f"{sample_id}.json"
        if not meta_path.exists():
            # Fall back to A4 metadata_path if relative.
            meta_rel = str(rec.get("metadata_path", ""))
            candidate = (a4_dir / meta_rel).resolve() if meta_rel else None
            if candidate is not None and candidate.exists():
                meta_path = candidate
            else:
                failures.append(
                    {"sample_id": sample_id, "reason": f"missing metadata: {meta_path}"}
                )
                continue

        metadata = load_sample_metadata(meta_path)
        metadata.setdefault("sample_id", sample_id)
        prompt_text = build_a5_prime_prompt(metadata, sample_id)
        prompt_name = prompt_filename_for_sample(sample_id)
        prompt_path = prompts_dir / prompt_name
        prompt_path.write_text(prompt_text, encoding="utf-8")

        # Metadata path: reuse verification_dataset relative link like A4.
        metadata_rel = f"../../verification_dataset/metadata/{sample_id}.json"
        rows.append(
            {
                "sample_id": sample_id,
                "condition": CONDITION_NAME,
                "prompt_path": f"prompts/{prompt_name}",
                "image_path": out_image_rel,
                "metadata_path": metadata_rel,
            }
        )

    n_validated = sum(1 for r in per_sample if r["identity_ok"])
    n_failed = len(failures)
    all_pass = n_failed == 0 and n_validated == args.sample_count

    # Aggregate hash of concatenated per-sample right-panel hashes (order = index).
    concat = "".join(r["sha256_right_panel"] for r in per_sample if r["identity_ok"])
    aggregate_hash = hashlib.sha256(concat.encode("utf-8")).hexdigest() if concat else None

    report = {
        "condition": CONDITION_NAME,
        "definition": "A5_prime = A4[:, width//2:]",
        "a4_dir": str(a4_dir),
        "output_dir": str(out_dir),
        "expected_n": args.sample_count,
        "n_validated": n_validated,
        "n_failed": n_failed,
        "all_pass": all_pass,
        "canonical_a5_prime_shape": list(expected_shape) if expected_shape else None,
        "aggregate_sha256_of_panel_hashes": aggregate_hash,
        "failures": failures,
        # Keep full per-sample detail for audit; large but required for 1000 gate.
        "per_sample": per_sample,
    }
    report_path = out_dir / "PIXEL_IDENTITY_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not all_pass:
        print("PIXEL IDENTITY HARD FAIL")
        print(f"  validated: {n_validated}/{args.sample_count}")
        print(f"  failed:    {n_failed}")
        print(f"  report:    {report_path}")
        for item in failures[:10]:
            print(f"  - {item}")
        return 2

    prompt_index = pd.DataFrame(rows)
    prompt_index.to_csv(out_dir / "prompt_index.csv", index=False)

    smoke = prompt_index[prompt_index["sample_id"].isin(SMOKE20_SAMPLE_IDS)].copy()
    # Preserve robustness manifest order.
    smoke["_ord"] = smoke["sample_id"].map({s: i for i, s in enumerate(SMOKE20_SAMPLE_IDS)})
    smoke = smoke.sort_values("_ord").drop(columns=["_ord"])
    if len(smoke) != 20:
        print(
            f"ERROR: smoke20 expected 20 rows, got {len(smoke)}",
            file=sys.stderr,
        )
        return 3
    smoke.to_csv(out_dir / "prompt_index_smoke20.csv", index=False)

    print("PIXEL IDENTITY PASS")
    print(f"  N validated:     {n_validated}")
    print(f"  N failed:        {n_failed}")
    print(f"  dimensions:      {expected_shape}")
    print(f"  aggregate_sha256:{aggregate_hash}")
    print(f"  prompt_index:    {out_dir / 'prompt_index.csv'}")
    print(f"  smoke20 index:   {out_dir / 'prompt_index_smoke20.csv'}")
    print(f"  report:          {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
