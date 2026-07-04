#!/usr/bin/env python3
"""
Purpose:
    Summarize Qwen verification ablation results across all conditions.

Input:
    - outputs/verification_ablation_results/{condition}/sample_*.json

Output:
    - outputs/verification_ablation_results/ablation_summary.csv
    - outputs/verification_ablation_results/ablation_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import VERIFICATION_ABLATION_RESULTS_DIR

DECISION_LABELS = ("Reliable", "Uncertain", "Unreliable")

SUMMARY_COLUMNS = [
    "condition",
    "total_samples",
    "successful_inference",
    "parse_failures",
    "inference_failures",
    "reliable_count",
    "uncertain_count",
    "unreliable_count",
    "reliable_pct",
    "uncertain_pct",
    "unreliable_pct",
    "avg_runtime_seconds",
    "median_runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize verification ablation inference results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=VERIFICATION_ABLATION_RESULTS_DIR,
        help="Root directory containing per-condition result folders",
    )
    return parser.parse_args()


def discover_condition_dirs(results_root: Path) -> list[Path]:
    """Return condition subdirectories that contain sample result JSON files."""
    if not results_root.exists():
        return []

    condition_dirs: list[Path] = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir():
            continue
        if any(path.glob("sample_*.json")):
            condition_dirs.append(path)
    return condition_dirs


def load_result_record(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def normalize_decision(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in DECISION_LABELS:
        return text
    mapped = text.lower()
    for label in DECISION_LABELS:
        if mapped == label.lower():
            return label
    return ""


def classify_record(record: dict[str, Any]) -> str:
    """Return one of: success, parse_failure, inference_failure."""
    inference_error = str(record.get("inference_error", "") or "").strip()
    if inference_error:
        return "inference_failure"

    parse_error = str(record.get("parse_error", "") or "").strip()
    decision = normalize_decision(record.get("decision"))
    if parse_error or not decision:
        return "parse_failure"

    return "success"


def summarize_condition(condition_dir: Path) -> dict[str, Any]:
    """Compute summary statistics for one ablation condition."""
    records = [load_result_record(path) for path in sorted(condition_dir.glob("sample_*.json"))]

    total = len(records)
    inference_failures = parse_failures = successful = 0
    decision_counts = {label: 0 for label in DECISION_LABELS}
    runtimes: list[float] = []

    for record in records:
        status = classify_record(record)
        if status == "inference_failure":
            inference_failures += 1
        elif status == "parse_failure":
            parse_failures += 1
        else:
            successful += 1
            decision = normalize_decision(record.get("decision"))
            if decision in decision_counts:
                decision_counts[decision] += 1

        runtime = record.get("runtime_seconds")
        if runtime is not None:
            try:
                runtimes.append(float(runtime))
            except (TypeError, ValueError):
                pass

    def pct(count: int) -> float:
        if total == 0:
            return 0.0
        return round(100.0 * count / total, 2)

    runtime_series = pd.Series(runtimes, dtype="float64")

    return {
        "condition": condition_dir.name,
        "total_samples": total,
        "successful_inference": successful,
        "parse_failures": parse_failures,
        "inference_failures": inference_failures,
        "reliable_count": decision_counts["Reliable"],
        "uncertain_count": decision_counts["Uncertain"],
        "unreliable_count": decision_counts["Unreliable"],
        "reliable_pct": pct(decision_counts["Reliable"]),
        "uncertain_pct": pct(decision_counts["Uncertain"]),
        "unreliable_pct": pct(decision_counts["Unreliable"]),
        "avg_runtime_seconds": round(float(runtime_series.mean()), 4) if not runtime_series.empty else 0.0,
        "median_runtime_seconds": round(float(runtime_series.median()), 4) if not runtime_series.empty else 0.0,
    }


def build_summary_dataframe(condition_dirs: list[Path]) -> pd.DataFrame:
    rows = [summarize_condition(path) for path in condition_dirs]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summary_to_markdown(summary_df: pd.DataFrame) -> str:
    """Render summary DataFrame as a markdown table."""
    if summary_df.empty:
        return "# Verification Ablation Summary\n\nNo condition results found.\n"

    display_df = summary_df.copy()
    for column in ("reliable_pct", "uncertain_pct", "unreliable_pct"):
        display_df[column] = display_df[column].map(lambda v: f"{v:.2f}%")

    lines = [
        "# Verification Ablation Summary",
        "",
        display_df.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_root = args.results_dir.resolve()

    condition_dirs = discover_condition_dirs(results_root)
    if not condition_dirs:
        print(f"No ablation condition folders found under: {results_root}")
        print("Expected subdirectories containing sample_*.json files.")
        sys.exit(1)

    summary_df = build_summary_dataframe(condition_dirs)

    csv_path = results_root / "ablation_summary.csv"
    md_path = results_root / "ablation_summary.md"
    markdown_text = summary_to_markdown(summary_df)

    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_text, encoding="utf-8")

    print("Verification ablation analysis")
    print(f"  Results root: {results_root}")
    print(f"  Conditions:   {len(condition_dirs)}")
    print()
    print(markdown_text)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")


if __name__ == "__main__":
    main()
