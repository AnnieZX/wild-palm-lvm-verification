"""LLaVA-OneVision verification adapter for the frozen verification framework."""

from __future__ import annotations

import time
import traceback

from src.lvm.llava_verifier import LlavaVerifier
from src.lvm.verification_response_parser import parse_verification_response
from src.verification.base_adapter import BaseVerificationAdapter, VerificationOutcome
from src.verification.jobs import VerificationJob
from src.verification.records import build_result_record


def build_llava_adapter(
    *,
    model_name: str,
    batch_size: int = 4,
    device_map: str = "auto",
    max_new_tokens: int = 512,
    model_key: str = "llava",
    condition: str = "",
    experiment_id: str = "",
) -> "LlavaVerificationAdapter":
    """Factory used by the verification adapter registry."""
    return LlavaVerificationAdapter(
        model_name=model_name,
        batch_size=batch_size,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        model_key=model_key,
        condition=condition,
        experiment_id=experiment_id,
    )


class LlavaVerificationAdapter(BaseVerificationAdapter):
    """
    LLaVA-OneVision verification adapter.

    Implements model-specific inference only. Orchestration, resume, and
    output persistence are handled by VerificationRunner.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        model_key: str = "llava",
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
        self._verifier = LlavaVerifier(model_name=model_name, device_map=device_map)

    @property
    def model_label(self) -> str:
        return "LLaVA-OneVision"

    def verify(self, job: VerificationJob) -> VerificationOutcome:
        """Run LLaVA-OneVision inference for one verification sample."""
        started_at = time.perf_counter()
        metadata = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "condition": self.condition,
            "experiment_id": self.experiment_id,
        }
        try:
            prompt = job.prompt_path.read_text(encoding="utf-8")
            raw_response = self._verifier.generate_response(
                image_path=job.image_path,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
            )
            runtime_seconds = time.perf_counter() - started_at

            parsed = None
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
