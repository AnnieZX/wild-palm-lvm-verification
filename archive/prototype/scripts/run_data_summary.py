#!/usr/bin/env python3
"""Analyze palm annotations and write summary statistics to CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.palm_analyzer import extract_all_palms

JSON_DIR = PROJECT_ROOT / "data" / "samples" / "json"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "data_summary.csv"


def palms_to_dataframe(palms) -> pd.DataFrame:
    """Convert palm instances into a flat pandas DataFrame."""
    rows = []
    for palm in palms:
        row = {
            "image": palm.image_name,
            "group_id": palm.group_id,
            "num_endpoints": palm.num_endpoints,
            "bbox_width": palm.bbox_width,
            "bbox_height": palm.bbox_height,
            "bbox_area": palm.bbox_area,
            "center_x": palm.center_x,
            "center_y": palm.center_y,
            "dist_mean": palm.dist_mean,
            "dist_min": palm.dist_min,
            "dist_max": palm.dist_max,
            "endpoint_distances": ";".join(
                f"{distance:.2f}" for distance in palm.endpoint_distances
            ),
        }
        for index, distance in enumerate(palm.endpoint_distances, start=1):
            row[f"dist_to_end_{index}"] = distance
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, num_images: int) -> None:
    """Print high-level statistics to the terminal."""
    num_palms = len(df)
    avg_endpoints = df["num_endpoints"].mean()
    avg_width = df["bbox_width"].mean()
    avg_height = df["bbox_height"].mean()
    avg_area = df["bbox_area"].mean()

    print(f"Images: {num_images}")
    print(f"Palms: {num_palms}")
    print(f"Average endpoints per palm: {avg_endpoints:.2f}")
    print()
    print("Average bbox size:")
    print(f"  width:  {avg_width:.2f} px")
    print(f"  height: {avg_height:.2f} px")
    print(f"  area:   {avg_area:.2f} px²")
    print()
    print("Bbox width (px):")
    print(f"  min: {df['bbox_width'].min():.2f}")
    print(f"  max: {df['bbox_width'].max():.2f}")
    print()
    print("Bbox height (px):")
    print(f"  min: {df['bbox_height'].min():.2f}")
    print(f"  max: {df['bbox_height'].max():.2f}")
    print()
    print("Bbox area (px²):")
    print(f"  min: {df['bbox_area'].min():.2f}")
    print(f"  max: {df['bbox_area'].max():.2f}")


def main() -> None:
    if not JSON_DIR.exists():
        print(f"JSON folder not found: {JSON_DIR}")
        sys.exit(1)

    json_files = sorted(JSON_DIR.glob("*.json"))
    palms = extract_all_palms(JSON_DIR)

    if not palms:
        print("No palm instances found.")
        sys.exit(1)

    df = palms_to_dataframe(palms)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(df)} palm record(s) to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print()
    print_summary(df, num_images=len(json_files))


if __name__ == "__main__":
    main()
