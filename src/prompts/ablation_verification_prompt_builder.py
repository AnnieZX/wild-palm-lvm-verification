"""Build verification ablation prompt/image inputs from the verification dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from src.preprocessing.ablation_verification_images import build_a4_combined_image
from src.prompts.ablation_verification_prompts import (
    ABLATION_CONDITIONS,
    build_ablation_verification_prompt,
)
from src.prompts.verification_prompt import prompt_filename_for_sample


def load_sample_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load one verification sample metadata JSON file."""
    with metadata_path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata object in {metadata_path}")
    return data


def _relative_path(from_dir: Path, target: Path) -> str:
    """Return a POSIX-style path relative to from_dir."""
    return os.path.relpath(target, from_dir).replace("\\", "/")


def build_condition_prompt_index(
    *,
    condition: str,
    condition_dir: Path,
    verification_dataset_dir: Path,
    samples_df: pd.DataFrame,
) -> pd.DataFrame:
    """Write prompts (and A4 images) for one ablation condition."""
    condition_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = condition_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    images_dir = condition_dir / "images"
    if condition == "A4_overlay_crop_confidence":
        images_dir.mkdir(parents=True, exist_ok=True)

    dataset_images_dir = verification_dataset_dir / "images"
    dataset_metadata_dir = verification_dataset_dir / "metadata"

    rows: list[dict[str, Any]] = []

    for _, record in samples_df.iterrows():
        sample_id = str(record["sample_id"])
        metadata_path = verification_dataset_dir / str(record["metadata_path"])
        overlay_path = verification_dataset_dir / str(record["image_path"])

        metadata = load_sample_metadata(metadata_path)
        metadata.setdefault("sample_id", sample_id)
        metadata.setdefault("confidence", record.get("confidence"))
        metadata.setdefault("bbox_width", record.get("bbox_width"))
        metadata.setdefault("bbox_height", record.get("bbox_height"))
        metadata.setdefault("bbox_area", record.get("bbox_area"))

        prompt_text = build_ablation_verification_prompt(metadata, condition)
        prompt_name = prompt_filename_for_sample(sample_id)
        prompt_path = prompts_dir / prompt_name
        prompt_path.write_text(prompt_text, encoding="utf-8")

        if condition == "A4_overlay_crop_confidence":
            overlay = cv2.imread(str(overlay_path))
            if overlay is None:
                raise FileNotFoundError(f"Could not read overlay image: {overlay_path}")

            bbox = metadata.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bbox = (
                    float(record["bbox_x"]),
                    float(record["bbox_y"]),
                    float(record["bbox_width"]),
                    float(record["bbox_height"]),
                )

            combined = build_a4_combined_image(overlay, tuple(float(v) for v in bbox))
            image_rel = f"images/{sample_id}.png"
            cv2.imwrite(str(condition_dir / image_rel), combined)
        else:
            image_rel = _relative_path(condition_dir, dataset_images_dir / f"{sample_id}.png")

        metadata_rel = _relative_path(condition_dir, dataset_metadata_dir / f"{sample_id}.json")

        rows.append(
            {
                "sample_id": sample_id,
                "condition": condition,
                "prompt_path": str((Path("prompts") / prompt_name).as_posix()),
                "image_path": image_rel,
                "metadata_path": metadata_rel,
            }
        )

    prompt_index = pd.DataFrame(rows)
    prompt_index_path = condition_dir / "prompt_index.csv"
    prompt_index.to_csv(prompt_index_path, index=False)
    return prompt_index


def build_ablation_verification_inputs(
    verification_dataset_dir: Path,
    output_root: Path,
    sample_count: int = 100,
    conditions: tuple[str, ...] = ABLATION_CONDITIONS,
) -> pd.DataFrame:
    """
    Build all verification ablation prompt/image inputs.

    Returns:
        Summary DataFrame written to output_root / ablation_prompt_summary.csv
    """
    verification_dataset_dir = verification_dataset_dir.resolve()
    output_root = output_root.resolve()
    index_csv = verification_dataset_dir / "index.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Verification dataset index not found: {index_csv}")

    samples_df = pd.read_csv(index_csv).head(sample_count)
    if samples_df.empty:
        raise ValueError("No verification samples available for ablation.")

    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for condition in conditions:
        condition_dir = output_root / condition
        prompt_index = build_condition_prompt_index(
            condition=condition,
            condition_dir=condition_dir,
            verification_dataset_dir=verification_dataset_dir,
            samples_df=samples_df,
        )

        if condition == "A4_overlay_crop_confidence":
            image_dir = str((condition_dir / "images").relative_to(output_root))
        else:
            image_dir = _relative_path(output_root, verification_dataset_dir / "images")

        summary_rows.append(
            {
                "condition": condition,
                "sample_count": len(prompt_index),
                "prompt_dir": str((condition_dir / "prompts").relative_to(output_root)),
                "image_dir": image_dir,
                "prompt_index": str((condition_dir / "prompt_index.csv").relative_to(output_root)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "ablation_prompt_summary.csv", index=False)
    return summary_df
