#!/usr/bin/env python3
"""
Generate publication-quality qualitative figures for Qwen verification experiments.

Input:
    - outputs/verification/qwen/<experiment_id>/A1..A5 inference results
    - outputs/evaluation/qwen/<experiment_id>/A1..A5 evaluation CSVs
    - outputs/verification_dataset/ images and index
    - Raw_Patches PNGs and LabelMe JSON

Output:
    - outputs/visualization/<experiment_id>/{overlay,comparison,failure_cases}/

Example:
    python scripts/visualization/visualize_verification.py \
        --experiment-id 20260706_2214 \
        --sample-count 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import qwen_experiment_visualization_dir
from src.visualization.experiment_data import (
    build_experiment_context,
    classify_failure_case,
    ensure_output_subdirs,
    find_best_ablation_code,
    load_primary_evaluation_pool,
    load_verification_result,
    row_to_sample_record,
    select_samples,
    verification_results_dir,
)
from src.visualization.figure_ablation import save_ablation_comparison
from src.visualization.figure_comparison import save_verification_comparison
from src.visualization.figure_failure import save_failure_case_figure
from src.visualization.figure_overlay import save_yolo_gt_overlay
from src.visualization.publication_style import apply_publication_style

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality verification visualization figures.",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Qwen experiment run id (e.g. 20260706_2214)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="Number of samples to visualize (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: outputs/visualization/<experiment_id>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--primary-ablation",
        type=str,
        default="A3",
        help="Reference ablation for evaluation pool (default: A3)",
    )
    parser.add_argument(
        "--ablation-sample-id",
        type=str,
        default=None,
        help="Sample id for five-ablation comparison (default: first selected sample)",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_record(
    context,
    row: pd.Series,
) -> "SampleRecord":
    result = load_verification_result(
        verification_results_dir(context, context.primary_ablation),
        str(row["sample_id"]),
    )
    return row_to_sample_record(row, result, context.primary_ablation)


def main() -> None:
    args = parse_args()
    configure_logging()
    apply_publication_style()

    started = time.perf_counter()
    output_dir = args.output_dir or qwen_experiment_visualization_dir(args.experiment_id)

    try:
        context = build_experiment_context(
            args.experiment_id,
            output_dir=output_dir,
            primary_ablation=args.primary_ablation,
        )
        pool = load_primary_evaluation_pool(context)
        selected = select_samples(pool, args.sample_count, args.seed)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        sys.exit(1)

    subdirs = ensure_output_subdirs(output_dir)
    best_ablation = find_best_ablation_code(context)

    overlay_count = 0
    comparison_count = 0
    ablation_count = 0
    failure_count = 0

    logger.info("Experiment: %s", context.experiment_id)
    logger.info("Output: %s", output_dir)
    logger.info("Selected %d samples (seed=%d)", len(selected), args.seed)
    logger.info("Best ablation by F1: %s", best_ablation)

    ablation_sample_id = args.ablation_sample_id or str(selected.iloc[0]["sample_id"])

    for _, row in selected.iterrows():
        record = build_record(context, row)
        sample_id = record.sample_id

        if save_yolo_gt_overlay(
            context,
            record,
            subdirs["overlay"] / f"{sample_id}_overlay.png",
        ):
            overlay_count += 1

        if save_verification_comparison(
            context,
            record,
            subdirs["comparison"] / f"{sample_id}_comparison.png",
        ):
            comparison_count += 1

    ablation_rows = selected[selected["sample_id"].astype(str) == ablation_sample_id]
    if not ablation_rows.empty:
        ablation_record = build_record(context, ablation_rows.iloc[0])
        if save_ablation_comparison(
            context,
            ablation_record,
            subdirs["comparison"] / f"{ablation_sample_id}_ablation.png",
            best_ablation_code=best_ablation,
        ):
            ablation_count += 1

    for _, row in pool.iterrows():
        record = build_record(context, row)
        failure_type = classify_failure_case(record)
        if failure_type is None:
            continue
        sample_id = record.sample_id
        if save_failure_case_figure(
            context,
            record,
            failure_type,
            subdirs["failure_cases"] / f"{sample_id}_{failure_type}.png",
        ):
            failure_count += 1

    elapsed = time.perf_counter() - started
    total_figures = overlay_count + comparison_count + ablation_count + failure_count

    print()
    print("=" * 60)
    print("Visualization summary")
    print("=" * 60)
    print(f"Experiment ID:       {context.experiment_id}")
    print(f"Output directory:  {output_dir}")
    print(f"Overlay figures:     {overlay_count}")
    print(f"Comparison figures:  {comparison_count}")
    print(f"Ablation figures:    {ablation_count}")
    print(f"Failure figures:     {failure_count}")
    print(f"Total figures:       {total_figures}")
    print(f"Runtime:             {elapsed:.1f}s")
    print()
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
