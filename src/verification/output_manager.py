"""Model-independent verification output persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.verification_resume import (
    RESULTS_INDEX_FILENAME,
    backfill_results_index_from_json,
    merge_results_index,
    write_results_index,
)


class VerificationOutputManager:
    """Write per-sample JSON results and results_index.csv."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir.resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.results_dir / RESULTS_INDEX_FILENAME

    def result_path(self, sample_id: str) -> Path:
        return self.results_dir / f"{sample_id}.json"

    def save_json(self, sample_id: str, record: dict[str, Any]) -> str:
        """Write one sample JSON and return its path relative to results_dir."""
        path = self.result_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
        return str(path.relative_to(self.results_dir))

    def finalize_index(self, new_rows: pd.DataFrame, *, resume: bool) -> pd.DataFrame:
        """Merge, backfill, and atomically write results_index.csv."""
        if resume:
            index_df = merge_results_index(self.index_path, new_rows)
            index_df = backfill_results_index_from_json(self.results_dir, index_df)
        else:
            index_df = new_rows.copy()

        write_results_index(self.index_path, index_df)
        return index_df
