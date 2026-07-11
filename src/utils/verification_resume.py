"""Resume helpers for verification inference runs.

Resume identity is logically ``(model_key, condition, sample_id)`` but is
implemented via directory isolation:

- Each model uses ``outputs/verification/<model_key>/<experiment_id>/<condition>/``
- Each condition has its own ``results_index.csv`` and ``sample_*.json`` set
- ``sample_id`` is unique within the verification dataset index

Therefore ``load_completed_sample_ids(results_dir)`` is safe as long as callers
pass the correct per-model, per-condition results directory. No cross-model
resume collision occurs when this convention is followed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_INDEX_FILENAME = "results_index.csv"
RESULTS_INDEX_COLUMNS = ("sample_id", "result_path", "status")


def load_completed_sample_ids(results_dir: Path) -> set[str]:
    """
    Return sample ids already completed in results_dir.

    Uses existing JSON outputs as the source of truth, supplemented by
    results_index.csv when present.
    """
    results_dir = results_dir.resolve()
    completed: set[str] = set()

    for json_path in results_dir.glob("sample_*.json"):
        completed.add(json_path.stem)

    index_path = results_dir / RESULTS_INDEX_FILENAME
    if index_path.exists():
        index_df = pd.read_csv(index_path)
        if "sample_id" in index_df.columns:
            completed.update(str(sample_id) for sample_id in index_df["sample_id"].dropna())

    return completed


def merge_results_index(existing_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to an existing results_index.csv without overwriting prior entries."""
    if new_rows.empty and not existing_path.exists():
        return pd.DataFrame(columns=list(RESULTS_INDEX_COLUMNS))

    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        combined = pd.concat([existing_df, new_rows], ignore_index=True)
        if "sample_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["sample_id"], keep="first")
        return combined

    return new_rows.copy()


def backfill_results_index_from_json(results_dir: Path, index_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every sample_*.json on disk has a row in results_index.csv."""
    results_dir = results_dir.resolve()
    if index_df.empty or "sample_id" not in index_df.columns:
        indexed_ids: set[str] = set()
        rows: list[dict[str, str]] = []
    else:
        indexed_ids = {str(sample_id) for sample_id in index_df["sample_id"].dropna()}
        rows = index_df.to_dict("records")

    for json_path in sorted(results_dir.glob("sample_*.json")):
        sample_id = json_path.stem
        if sample_id in indexed_ids:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "result_path": json_path.name,
                "status": "ok",
            }
        )
        indexed_ids.add(sample_id)

    if not rows:
        return pd.DataFrame(columns=list(RESULTS_INDEX_COLUMNS))

    backfilled = pd.DataFrame(rows)
    if "sample_id" in backfilled.columns:
        backfilled = backfilled.drop_duplicates(subset=["sample_id"], keep="first")
    return backfilled


def write_results_index(index_path: Path, index_df: pd.DataFrame) -> None:
    """Atomically write results_index.csv."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = index_path.with_suffix(".csv.tmp")
    index_df.to_csv(temp_path, index=False)
    temp_path.replace(index_path)
