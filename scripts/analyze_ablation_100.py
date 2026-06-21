#!/usr/bin/env python3
"""Summarize 100-palm ablation results."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

COMBINED_CSV = PROJECT_ROOT / "outputs" / "ablation_results_100_combined.csv"
METADATA_CSV = PROJECT_ROOT / "outputs" / "ablation_metadata_100.csv"
SUMMARY_CSV = PROJECT_ROOT / "outputs" / "ablation_summary_100.csv"
ENDPOINT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "ablation_endpoint_summary_100.csv"
ANCHORING_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "ablation_anchoring_summary_100.csv"

EXPECTED_COMBINED_ROWS = 1000


def endpoint_bin(count: float | int | None) -> str:
    if count is None or pd.isna(count):
        return "unknown"
    count = int(count)
    if count == 0:
        return "0"
    if 1 <= count <= 2:
        return "1-2"
    if 3 <= count <= 5:
        return "3-5"
    return ">5"


def yolo_bin(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    value = float(value)
    if value < 0.4:
        return "low"
    if value <= 0.7:
        return "medium"
    return "high"


def safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def safe_std(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.std())


def count_quality(df: pd.DataFrame, label: str) -> int:
    return int((df["detection_quality"] == label).sum())


def count_is_palm(df: pd.DataFrame, label: str) -> int:
    return int((df["is_palm"] == label).sum())


def build_condition_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for condition_name, group in combined_df.groupby("condition_name", sort=True):
        row = {
            "condition_name": condition_name,
            "n": len(group),
            "parse_error_count": int((group["parse_error"].astype(str).str.strip() != "").sum()),
            "inference_error_count": int(
                (group["inference_error"].astype(str).str.strip() != "").sum()
            ),
            "is_palm_yes_count": count_is_palm(group, "yes"),
            "is_palm_uncertain_count": count_is_palm(group, "uncertain"),
            "is_palm_no_count": count_is_palm(group, "no"),
            "reliable_count": count_quality(group, "reliable"),
            "uncertain_count": count_quality(group, "uncertain"),
            "unreliable_count": count_quality(group, "unreliable"),
            "mean_palm_confidence": safe_mean(group["palm_confidence"]),
            "std_palm_confidence": safe_std(group["palm_confidence"]),
            "mean_confidence": safe_mean(group["confidence"]),
            "std_confidence": safe_std(group["confidence"]),
        }
        row["parse_error_rate"] = row["parse_error_count"] / row["n"] if row["n"] else 0.0
        row["inference_error_rate"] = (
            row["inference_error_count"] / row["n"] if row["n"] else 0.0
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_endpoint_summary(combined_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    merged = combined_df.merge(
        metadata_df[["sample_index", "endpoints_count"]],
        on="sample_index",
        how="left",
    )
    merged["endpoint_bin"] = merged["endpoints_count"].apply(endpoint_bin)

    rows: list[dict] = []
    grouped = merged.groupby(["condition_name", "endpoint_bin"], sort=True)
    for (condition_name, endpoint_bin_label), group in grouped:
        total = len(group)
        rows.append(
            {
                "condition_name": condition_name,
                "endpoint_bin": endpoint_bin_label,
                "n": total,
                "reliable_count": count_quality(group, "reliable"),
                "uncertain_count": count_quality(group, "uncertain"),
                "unreliable_count": count_quality(group, "unreliable"),
                "reliable_rate": count_quality(group, "reliable") / total if total else 0.0,
                "uncertain_rate": count_quality(group, "uncertain") / total if total else 0.0,
                "unreliable_rate": count_quality(group, "unreliable") / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_anchoring_summary(combined_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    target_conditions = [
        name
        for name in combined_df["condition_name"].unique()
        if name.endswith("P4_yolo_confidence") or name.endswith("P6_full_metadata")
    ]

    subset = combined_df[combined_df["condition_name"].isin(target_conditions)].copy()
    merged = subset.merge(
        metadata_df[["sample_index", "yolo_confidence"]],
        on="sample_index",
        how="left",
    )
    merged["yolo_bin"] = merged["yolo_confidence"].apply(yolo_bin)

    rows: list[dict] = []
    grouped = merged.groupby(["condition_name", "yolo_bin"], sort=True)
    for (condition_name, yolo_bin_label), group in grouped:
        total = len(group)
        reliable = count_quality(group, "reliable")
        rows.append(
            {
                "condition_name": condition_name,
                "yolo_bin": yolo_bin_label,
                "n": total,
                "reliable_count": reliable,
                "proportion_reliable": reliable / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not COMBINED_CSV.exists():
        print(f"Combined results not found: {COMBINED_CSV}")
        print("Run run_qwen_ablation_100.py first.")
        sys.exit(1)

    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        sys.exit(1)

    combined_df = pd.read_csv(COMBINED_CSV)
    metadata_df = pd.read_csv(METADATA_CSV)

    if len(metadata_df) != 100:
        print(f"WARNING: metadata has {len(metadata_df)} rows (expected 100).")

    if len(combined_df) < EXPECTED_COMBINED_ROWS:
        print(
            f"WARNING: combined results has {len(combined_df)} rows "
            f"(expected {EXPECTED_COMBINED_ROWS})."
        )

    summary_df = build_condition_summary(combined_df)
    endpoint_df = build_endpoint_summary(combined_df, metadata_df)
    anchoring_df = build_anchoring_summary(combined_df, metadata_df)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    endpoint_df.to_csv(ENDPOINT_SUMMARY_CSV, index=False)
    anchoring_df.to_csv(ANCHORING_SUMMARY_CSV, index=False)

    print("Ablation analysis complete")
    print(f"  Combined rows analyzed: {len(combined_df)}")
    print(f"  Conditions: {combined_df['condition_name'].nunique()}")
    print()
    print("Output files:")
    print(f"  {METADATA_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {COMBINED_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {ENDPOINT_SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {ANCHORING_SUMMARY_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
