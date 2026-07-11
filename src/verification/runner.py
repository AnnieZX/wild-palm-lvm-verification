"""Model-independent verification inference runner."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.verification_resume import load_completed_sample_ids
from src.verification.base_adapter import BaseVerificationAdapter
from src.verification.jobs import VerificationJob, validate_jobs
from src.verification.output_manager import VerificationOutputManager


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime options for one verification run."""

    resume: bool = False
    skip_existing: bool = False
    experiment_id: str | None = None
    condition: str | None = None
    batch_size: int = 4


class VerificationRunner:
    """
    Orchestrate verification inference for any VLM adapter.

    Owns dataset iteration, resume filtering, progress logging, and output
    persistence. Adapters implement model-specific inference only.
    """

    def __init__(
        self,
        adapter: BaseVerificationAdapter,
        output_manager: VerificationOutputManager,
    ) -> None:
        self.adapter = adapter
        self.output_manager = output_manager

    def apply_resume(self, jobs: list[VerificationJob]) -> tuple[list[VerificationJob], int]:
        """Filter completed samples when resume is enabled."""
        completed_samples = load_completed_sample_ids(self.output_manager.results_dir)
        for job in jobs:
            if job.sample_id in completed_samples:
                print(f"Skipping sample {job.sample_id}")
        pending = [job for job in jobs if job.sample_id not in completed_samples]
        return pending, len(completed_samples)

    def print_resume_banner(
        self,
        *,
        completed_count: int,
        remaining: int,
        experiment_id: str | None,
        condition: str | None,
    ) -> None:
        print()
        print("----------------------------------------")
        print()
        print("Resume enabled")
        print()
        if experiment_id:
            print(f"Experiment:\n\n{experiment_id}")
            print()
        if condition:
            print(f"Condition:\n\n{condition}")
            print()
        print(f"Loaded {completed_count} completed samples")
        print()
        print(f"Remaining:\n\n{remaining}")
        print()
        print("----------------------------------------")
        print()

    def run(self, jobs: list[VerificationJob], config: RunnerConfig) -> pd.DataFrame:
        """Run verification for all pending jobs."""
        pending = jobs

        if config.resume:
            pending, completed_count = self.apply_resume(jobs)
            self.print_resume_banner(
                completed_count=completed_count,
                remaining=len(pending),
                experiment_id=config.experiment_id,
                condition=config.condition,
            )
            if not pending:
                print("All samples already completed. Nothing to do.")
                if self.output_manager.index_path.exists():
                    print(f"Existing index: {self.output_manager.index_path}")
                return self.output_manager.finalize_index(
                    pd.DataFrame(),
                    resume=True,
                )

        validate_jobs(pending)

        summary_rows: list[dict[str, str]] = []
        skip_existing = config.skip_existing or config.resume
        to_run: list[VerificationJob] = []

        for job in pending:
            if skip_existing and self.output_manager.result_path(job.sample_id).exists():
                summary_rows.append(
                    {
                        "sample_id": job.sample_id,
                        "result_path": f"{job.sample_id}.json",
                        "status": "skipped",
                    }
                )
                continue
            to_run.append(job)

        batch_size = max(1, config.batch_size)
        for batch_start in range(0, len(to_run), batch_size):
            batch_jobs = to_run[batch_start : batch_start + batch_size]
            print(
                f"\n--- Batch {batch_start // batch_size + 1}: "
                f"samples {batch_start + 1}-{batch_start + len(batch_jobs)} "
                f"of {len(to_run)} ---"
            )

            for job in batch_jobs:
                print(f"Running sample {job.sample_id}")
                outcome = self.adapter.verify(job)
                result_path = self.output_manager.save_json(job.sample_id, outcome.record)
                summary_rows.append(
                    {
                        "sample_id": job.sample_id,
                        "result_path": result_path,
                        "status": outcome.status,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        return self.output_manager.finalize_index(summary_df, resume=config.resume)
