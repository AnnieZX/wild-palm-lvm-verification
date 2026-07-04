"""Write verification prompts for every sample in a verification dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.prompts.verification_prompt import build_verification_detection_prompt, prompt_filename_for_sample


def load_sample_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load one verification sample metadata JSON file."""
    with metadata_path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata object in {metadata_path}")
    return data


def build_prompts_for_dataset(
    dataset_dir: Path,
    prompts_dir: Path | None = None,
    index_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Build one text prompt per verification sample.

    Reads overlay metadata from dataset_dir/metadata/ (via index.csv) and writes
    plain-text prompts to dataset_dir/prompts/ by default.

    Returns:
        DataFrame with sample_id, prompt_path, image_path, metadata_path.
    """
    dataset_dir = dataset_dir.resolve()
    prompts_dir = (prompts_dir or dataset_dir / "prompts").resolve()
    index_csv = index_csv or dataset_dir / "index.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    index_df = pd.read_csv(index_csv)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for _, record in index_df.iterrows():
        sample_id = str(record["sample_id"])
        metadata_rel = str(record["metadata_path"])
        image_rel = str(record["image_path"])
        metadata_path = dataset_dir / metadata_rel

        metadata = load_sample_metadata(metadata_path)
        prompt_text = build_verification_detection_prompt(metadata)

        prompt_name = prompt_filename_for_sample(sample_id)
        prompt_path = prompts_dir / prompt_name
        prompt_path.write_text(prompt_text, encoding="utf-8")

        rows.append(
            {
                "sample_id": sample_id,
                "prompt_path": str(prompt_path.relative_to(dataset_dir)),
                "image_path": image_rel,
                "metadata_path": metadata_rel,
            }
        )

    return pd.DataFrame(rows)
