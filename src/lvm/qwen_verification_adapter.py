"""Batch Qwen2.5-VL inference adapter for the verification dataset."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from src.lvm.qwen_verifier import QwenVerifier
from src.lvm.verification_response_parser import parse_verification_response
from src.verification.base_adapter import BaseVerificationAdapter, VerificationOutcome
from src.verification.jobs import VerificationJob, load_verification_jobs, validate_jobs
from src.verification.output_manager import VerificationOutputManager
from src.verification.records import build_result_record
from src.verification.runner import RunnerConfig, VerificationRunner

# Backward-compatible re-exports for existing imports.
__all__ = [
    "QwenVerificationAdapter",
    "VerificationJob",
    "build_qwen_adapter",
    "build_result_record",
    "load_verification_jobs",
    "save_result_record",
    "validate_jobs",
]


def save_result_record(result_path: Path, record: dict[str, Any]) -> None:
    """Backward-compatible JSON writer."""
    VerificationOutputManager(result_path.parent).save_json(
        str(record.get("sample_id", result_path.stem)),
        record,
    )


def build_qwen_adapter(
    *,
    model_name: str,
    batch_size: int = 4,
    device_map: str = "auto",
    max_new_tokens: int = 512,
    model_key: str = "qwen2_5_vl",
    condition: str = "",
    experiment_id: str = "",
) -> "QwenVerificationAdapter":
    """Factory used by the verification adapter registry."""
    return QwenVerificationAdapter(
        model_name=model_name,
        batch_size=batch_size,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        model_key=model_key,
        condition=condition,
        experiment_id=experiment_id,
    )


class QwenVerificationAdapter(BaseVerificationAdapter):
    """
    Qwen2.5-VL verification adapter.

    Implements model-specific inference only. Batch orchestration, resume, and
    output persistence are handled by VerificationRunner.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        model_key: str = "qwen2_5_vl",
        condition: str = "",
        experiment_id: str = "",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.model_key = model_key
        self.condition = condition
        self.experiment_id = experiment_id
        self._verifier = QwenVerifier(model_name=model_name, device_map=device_map)

    @property
    def model_label(self) -> str:
        return "Qwen2.5-VL"

    def verify(self, job: VerificationJob) -> VerificationOutcome:
        """Run Qwen inference for one verification sample."""
        started_at = time.perf_counter()
        metadata = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "condition": self.condition,
            "experiment_id": self.experiment_id,
        }
        try:
            prompt = job.prompt_path.read_text(encoding="utf-8")
            raw_response = self._verifier.generate_response_debugged(
                sample_id=job.sample_id,
                image_path=job.image_path,
                prompt_path=job.prompt_path,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
            )
            runtime_seconds = time.perf_counter() - started_at

            parsed: dict[str, str] | None = None
            parse_error = ""
            try:
                parsed = parse_verification_response(raw_response, model_key=self.model_key)
            except ValueError as error:
                parse_error = str(error)
                print(
                    f"PARSE ERROR for {job.sample_id}: {error}\n"
                    f"{traceback.format_exc()}"
                )

            record = build_result_record(
                sample_id=job.sample_id,
                raw_response=raw_response,
                parsed=parsed,
                runtime_seconds=runtime_seconds,
                parse_error=parse_error,
                **metadata,
            )
            status = "ok" if not parse_error else "parse_error"
            return VerificationOutcome(record=record, status=status)

        except Exception as error:
            runtime_seconds = time.perf_counter() - started_at
            tb = traceback.format_exc()
            print(
                f"\nINFERENCE ERROR for {job.sample_id}: "
                f"{type(error).__name__}: {error}\n{tb}",
                flush=True,
            )
            traceback.print_exc()

            record = build_result_record(
                sample_id=job.sample_id,
                raw_response="",
                parsed=None,
                runtime_seconds=runtime_seconds,
                inference_error=tb,
                **metadata,
            )
            return VerificationOutcome(record=record, status="inference_error")

    def run_jobs(
        self,
        jobs: list[VerificationJob],
        results_dir: Path,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """
        Backward-compatible batch entry point.

        Delegates orchestration to VerificationRunner.
        """
        runner = VerificationRunner(self, VerificationOutputManager(results_dir))
        return runner.run(
            jobs,
            RunnerConfig(skip_existing=skip_existing, batch_size=self.batch_size),
        )
