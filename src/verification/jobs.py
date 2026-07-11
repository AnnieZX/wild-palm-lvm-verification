"""Verification job loading utilities (model-independent)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class VerificationJob:
    """One verification sample ready for inference."""

    sample_id: str
    image_path: Path
    prompt_path: Path


def load_verification_jobs(
    dataset_dir: Path | None = None,
    prompt_index: Path | None = None,
) -> list[VerificationJob]:
    """
    Load verification jobs from a prompt index CSV.

    Args:
        dataset_dir: Root directory for resolving relative image/prompt paths.
            Used when prompt_index is not provided.
        prompt_index: Explicit path to prompt_index.csv. Paths in the CSV are
            resolved relative to its parent directory.
    """
    if prompt_index is not None:
        prompt_index = prompt_index.resolve()
        if not prompt_index.exists():
            raise FileNotFoundError(f"Prompt index not found: {prompt_index}")
        dataset_dir = prompt_index.parent
        index_df = pd.read_csv(prompt_index)
    else:
        if dataset_dir is None:
            raise ValueError("Either dataset_dir or prompt_index must be provided.")

        dataset_dir = dataset_dir.resolve()
        prompt_index_csv = dataset_dir / "prompt_index.csv"
        index_csv = dataset_dir / "index.csv"

        if prompt_index_csv.exists():
            index_df = pd.read_csv(prompt_index_csv)
        elif index_csv.exists():
            index_df = pd.read_csv(index_csv)
            index_df["prompt_path"] = index_df["sample_id"].map(lambda sid: f"prompts/{sid}.txt")
        else:
            raise FileNotFoundError(
                f"No index found under {dataset_dir}. Expected prompt_index.csv or index.csv."
            )

    jobs: list[VerificationJob] = []
    for _, row in index_df.iterrows():
        sample_id = str(row["sample_id"])
        image_path = (dataset_dir / str(row["image_path"])).resolve()
        prompt_path = (dataset_dir / str(row["prompt_path"])).resolve()
        jobs.append(
            VerificationJob(
                sample_id=sample_id,
                image_path=image_path,
                prompt_path=prompt_path,
            )
        )

    return jobs


def validate_jobs(jobs: list[VerificationJob]) -> None:
    """Ensure every job has readable image and prompt files."""
    missing: list[str] = []
    for job in jobs:
        if not job.image_path.exists():
            missing.append(str(job.image_path))
        if not job.prompt_path.exists():
            missing.append(str(job.prompt_path))

    if missing:
        sample = missing[:5]
        raise FileNotFoundError(
            "Missing verification input files:\n  "
            + "\n  ".join(sample)
            + (f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )
