"""Batch Qwen2.5-VL inference adapter for the verification dataset."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.lvm.qwen_verifier import QwenVerifier
from src.lvm.verification_response_parser import parse_verification_response


@dataclass(frozen=True)
class VerificationJob:
    """One verification sample ready for inference."""

    sample_id: str
    image_path: Path
    prompt_path: Path


def load_verification_jobs(dataset_dir: Path) -> list[VerificationJob]:
    """Load verification jobs from prompt_index.csv or index.csv."""
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
        image_path = dataset_dir / str(row["image_path"])
        prompt_path = dataset_dir / str(row["prompt_path"])
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


def build_result_record(
    sample_id: str,
    raw_response: str,
    parsed: dict[str, str] | None,
    runtime_seconds: float,
    parse_error: str = "",
) -> dict[str, Any]:
    """Build one verification result JSON record."""
    return {
        "sample_id": sample_id,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "decision": parsed.get("decision", "") if parsed else "",
        "confidence_reasoning": parsed.get("confidence_reasoning", "") if parsed else "",
        "visual_reasoning": parsed.get("visual_reasoning", "") if parsed else "",
        "runtime_seconds": round(runtime_seconds, 4),
        "parse_error": parse_error,
    }


def save_result_record(result_path: Path, record: dict[str, Any]) -> None:
    """Write one sample result JSON."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)


class QwenVerificationAdapter:
    """
    Run batched Qwen2.5-VL inference over a verification dataset.

    Model path must be supplied explicitly or loaded from configs/model.yaml.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        device_map: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._verifier = QwenVerifier(model_name=model_name, device_map=device_map)

    def run_jobs(
        self,
        jobs: list[VerificationJob],
        results_dir: Path,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """
        Run inference for all jobs and write per-sample JSON results.

        Returns:
            DataFrame summarizing each sample result path and status.
        """
        results_dir = results_dir.resolve()
        results_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        pending: list[VerificationJob] = []

        for job in jobs:
            result_path = results_dir / f"{job.sample_id}.json"
            if skip_existing and result_path.exists():
                summary_rows.append(
                    {
                        "sample_id": job.sample_id,
                        "result_path": str(result_path.relative_to(results_dir)),
                        "status": "skipped",
                    }
                )
                continue
            pending.append(job)

        for batch_start in range(0, len(pending), self.batch_size):
            batch_jobs = pending[batch_start : batch_start + self.batch_size]
            batch_items = [
                (job.image_path, job.prompt_path.read_text(encoding="utf-8"))
                for job in batch_jobs
            ]

            batch_started = time.perf_counter()
            try:
                raw_responses = self._verifier.generate_batch_responses(
                    batch_items,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as error:
                per_sample_runtime = 0.0
                for job in batch_jobs:
                    record = build_result_record(
                        sample_id=job.sample_id,
                        raw_response="",
                        parsed=None,
                        runtime_seconds=per_sample_runtime,
                    )
                    result_path = results_dir / f"{job.sample_id}.json"
                    save_result_record(result_path, record)
                    summary_rows.append(
                        {
                            "sample_id": job.sample_id,
                            "result_path": str(result_path.relative_to(results_dir)),
                            "status": "inference_error",
                        }
                    )
                continue

            batch_runtime = time.perf_counter() - batch_started
            per_sample_runtime = batch_runtime / len(batch_jobs)

            for job, raw_response in zip(batch_jobs, raw_responses):
                parsed: dict[str, str] | None = None
                parse_error = ""
                try:
                    parsed = parse_verification_response(raw_response)
                except ValueError as error:
                    parse_error = str(error)

                record = build_result_record(
                    sample_id=job.sample_id,
                    raw_response=raw_response,
                    parsed=parsed,
                    runtime_seconds=per_sample_runtime,
                    parse_error=parse_error,
                )
                result_path = results_dir / f"{job.sample_id}.json"
                save_result_record(result_path, record)
                summary_rows.append(
                    {
                        "sample_id": job.sample_id,
                        "result_path": str(result_path.relative_to(results_dir)),
                        "status": "ok" if not parse_error else "parse_error",
                    }
                )

        return pd.DataFrame(summary_rows)
