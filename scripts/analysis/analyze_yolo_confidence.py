#!/usr/bin/env python3
"""
Analyze YOLO confidence scores for detections used in the verification pipeline.

Loads detections from the verification dataset index when available, otherwise
replicates the dataset builder filter (confidence threshold + resolvable images).

Outputs summary statistics, percentile table, and publication-style figures under
outputs/analysis/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.gt_matching import greedy_match_detections_to_gt
from src.paths import (
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.preprocessing.gt_palm_bboxes import extract_gt_palm_bboxes
from src.preprocessing.verification_dataset import (
    build_png_index,
    iter_verification_samples,
    resolve_patch_image_from_index,
)
from src.utils.labelme_paths import resolve_labelme_json
from src.yolo.predictions_io import load_predictions

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
FIGURE_DPI = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze YOLO confidence distribution for verification-pipeline detections."
        ),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=VERIFICATION_DATASET_INDEX_CSV,
        help="Verification dataset index.csv (default: outputs/verification_dataset/index.csv)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_FULL_JSON,
        help="YOLO predictions JSON (used when index.csv is missing)",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="Raw patch PNG root for fallback prediction filtering",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="LabelMe JSON root for optional TP/FP analysis",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence when rebuilding from predictions (default: 0.5)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for TP/FP matching (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and summary files",
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        help="Skip TP/FP confidence analysis even if annotations are available",
    )
    return parser.parse_args()


def load_verification_detections(
    index_csv: Path,
    predictions_path: Path,
    images_root: Path,
    confidence_threshold: float,
) -> tuple[pd.DataFrame, str]:
    """
    Return verification-pipeline detections and a short source description.

    Prefers index.csv when present; otherwise filters predictions the same way
    as generate_verification_dataset.py (threshold + resolvable image).
    """
    if index_csv.is_file():
        df = pd.read_csv(index_csv)
        required = {"sample_id", "image_name", "confidence", "bbox_x", "bbox_y", "bbox_width", "bbox_height"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"index.csv missing columns: {sorted(missing)}")
        return df, f"verification dataset index ({index_csv})"

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Neither index.csv ({index_csv}) nor predictions ({predictions_path}) found."
        )

    records = load_predictions(predictions_path)
    samples = iter_verification_samples(records, confidence_threshold)
    png_index = build_png_index(images_root)

    rows: list[dict[str, Any]] = []
    sample_counter = 0
    for image_id, prediction in samples:
        if resolve_patch_image_from_index(images_root, image_id, png_index) is None:
            continue

        sample_counter += 1
        x, y, width, height = prediction["bbox"]
        confidence = prediction["score"]
        rows.append(
            {
                "sample_id": f"sample_{sample_counter:06d}",
                "image_name": image_id,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "confidence": confidence,
            }
        )

    if not rows:
        raise ValueError(
            "No verification detections found after filtering predictions. "
            "Generate the verification dataset first or check paths."
        )

    return pd.DataFrame(rows), f"filtered predictions ({predictions_path})"


def compute_summary_stats(confidences: np.ndarray) -> dict[str, float]:
    """Compute distribution summary for one confidence array."""
    if confidences.size == 0:
        return {
            "total_detections": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "total_detections": int(confidences.size),
        "mean": float(np.mean(confidences)),
        "median": float(np.median(confidences)),
        "std": float(np.std(confidences, ddof=0)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
    }


def compute_percentiles(confidences: np.ndarray) -> dict[str, float]:
    """Return named percentile values."""
    if confidences.size == 0:
        return {f"p{p:02d}": float("nan") for p in PERCENTILES}

    values = np.percentile(confidences, PERCENTILES)
    return {f"p{p:02d}": float(v) for p, v in zip(PERCENTILES, values)}


def kde_is_appropriate(confidences: np.ndarray) -> bool:
    """Heuristic: KDE needs enough samples and non-zero spread."""
    if confidences.size < 30:
        return False
    if float(np.std(confidences)) < 1e-6:
        return False
    return True


def gaussian_kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Simple Gaussian KDE using NumPy only (Silverman bandwidth)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    std = float(values.std(ddof=1)) if n > 1 else 1.0
    bandwidth = std * (n ** (-1.0 / 5.0))
    if bandwidth <= 0:
        return np.zeros_like(grid)

    diff = grid[:, None] - values[None, :]
    kernel = np.exp(-0.5 * (diff / bandwidth) ** 2) / (bandwidth * np.sqrt(2.0 * np.pi))
    return kernel.mean(axis=1)


def classify_tp_fp(
    detections_df: pd.DataFrame,
    annotations_root: Path,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Label each verification detection as TP or FP using greedy IoU matching.

    Returns (tp_confidences, fp_confidences, gt_available).
    """
    if not annotations_root.exists():
        return np.array([]), np.array([]), False

    tp_scores: list[float] = []
    fp_scores: list[float] = []
    resolved_images = 0

    for image_name, group in detections_df.groupby("image_name", sort=False):
        json_path = resolve_labelme_json(annotations_root, str(image_name))
        if json_path is None or not json_path.is_file():
            continue

        resolved_images += 1
        gt_bboxes = extract_gt_palm_bboxes(json_path)
        detection_records: list[dict[str, Any]] = []

        for _, row in group.iterrows():
            detection_records.append(
                {
                    "bbox": (
                        float(row["bbox_x"]),
                        float(row["bbox_y"]),
                        float(row["bbox_width"]),
                        float(row["bbox_height"]),
                    ),
                    "score": float(row["confidence"]),
                }
            )

        _, matched_detections, _ = greedy_match_detections_to_gt(
            detection_records,
            gt_bboxes,
            iou_threshold,
        )

        for det_index, record in enumerate(detection_records):
            score = float(record["score"])
            if det_index in matched_detections:
                tp_scores.append(score)
            else:
                fp_scores.append(score)

    return np.asarray(tp_scores), np.asarray(fp_scores), resolved_images > 0


def save_statistics_csv(
    output_path: Path,
    overall: dict[str, float],
    percentiles: dict[str, float],
    tp_stats: dict[str, float] | None,
    fp_stats: dict[str, float] | None,
) -> None:
    """Write flat metric/value CSV."""
    rows: list[dict[str, str]] = []

    def append_group(prefix: str, stats: dict[str, float], pct: dict[str, float] | None = None) -> None:
        for key, value in stats.items():
            rows.append({"group": prefix, "metric": key, "value": value})
        if pct:
            for key, value in pct.items():
                rows.append({"group": prefix, "metric": key, "value": value})

    append_group("overall", overall, percentiles)
    if tp_stats is not None:
        append_group("true_positive", tp_stats)
    if fp_stats is not None:
        append_group("false_positive", fp_stats)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_histogram_and_kde(
    confidences: np.ndarray,
    output_path: Path,
    kde_ok: bool,
) -> None:
    """Figure 1 (histogram) and Figure 2 (KDE) in one file."""
    fig, axes = plt.subplots(2 if kde_ok else 1, 1, figsize=(8, 8 if kde_ok else 4), squeeze=False)
    ax_hist = axes[0][0]

    ax_hist.hist(
        confidences,
        bins=min(50, max(10, int(np.sqrt(confidences.size)))),
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
    )
    ax_hist.set_xlabel("YOLO confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Histogram of YOLO confidence scores (verification pipeline)")
    ax_hist.grid(axis="y", alpha=0.3)

    if kde_ok:
        ax_kde = axes[1][0]
        grid = np.linspace(confidences.min(), confidences.max(), 256)
        density = gaussian_kde_1d(confidences, grid)
        ax_kde.plot(grid, density, color="#C44E52", linewidth=2)
        ax_kde.fill_between(grid, density, alpha=0.25, color="#C44E52")
        ax_kde.set_xlabel("YOLO confidence")
        ax_kde.set_ylabel("Density")
        ax_kde.set_title("Kernel density estimate")
        ax_kde.grid(alpha=0.3)
    else:
        fig.text(
            0.5,
            0.02,
            "KDE omitted: insufficient samples or zero variance.",
            ha="center",
            fontsize=9,
            color="0.4",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(confidences: np.ndarray, output_path: Path) -> None:
    """Figure 3: boxplot of confidence scores."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        confidences,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "#4C72B0", "alpha": 0.6},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
    )
    ax.set_ylabel("YOLO confidence")
    ax.set_title("Boxplot of YOLO confidence scores")
    ax.set_xticklabels(["Verification detections"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_tp_fp_comparison(
    tp_confidences: np.ndarray,
    fp_confidences: np.ndarray,
    output_path: Path,
) -> None:
    """Compare TP vs FP confidence distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = [tp_confidences, fp_confidences]
    labels = [
        f"True positive (n={tp_confidences.size})",
        f"False positive (n={fp_confidences.size})",
    ]
    colors = ["#55A868", "#C44E52"]

    axes[0].hist(
        data,
        bins=30,
        label=labels,
        color=colors,
        alpha=0.65,
        edgecolor="white",
    )
    axes[0].set_xlabel("YOLO confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("TP vs FP confidence (histogram)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    bp = axes[1].boxplot(
        data,
        labels=["TP", "FP"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("YOLO confidence")
    axes[1].set_title("TP vs FP confidence (boxplot)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Detection confidence by greedy GT match (IoU ≥ 0.5)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    source: str,
    overall: dict[str, float],
    percentiles: dict[str, float],
    tp_stats: dict[str, float] | None,
    fp_stats: dict[str, float] | None,
) -> None:
    """Print human-readable summary to stdout."""
    print()
    print("YOLO confidence analysis (verification pipeline)")
    print(f"  Source: {source}")
    print()
    print("Summary statistics")
    print(f"  Total detections: {overall['total_detections']}")
    print(f"  Mean:             {overall['mean']:.4f}")
    print(f"  Median:           {overall['median']:.4f}")
    print(f"  Std dev:          {overall['std']:.4f}")
    print(f"  Min:              {overall['min']:.4f}")
    print(f"  Max:              {overall['max']:.4f}")
    print()
    print("Percentiles")
    for percentile in PERCENTILES:
        key = f"p{percentile:02d}"
        print(f"  {percentile:>3}%: {percentiles[key]:.4f}")

    if tp_stats is not None and fp_stats is not None:
        print()
        print("GT matching (IoU ≥ 0.5)")
        print(f"  TP detections:    {tp_stats['total_detections']}")
        print(f"  TP mean conf:     {tp_stats['mean']:.4f}")
        print(f"  FP detections:    {fp_stats['total_detections']}")
        print(f"  FP mean conf:     {fp_stats['mean']:.4f}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detections_df, source = load_verification_detections(
        index_csv=args.index_csv,
        predictions_path=args.predictions,
        images_root=args.images_root,
        confidence_threshold=args.confidence_threshold,
    )
    confidences = detections_df["confidence"].astype(float).to_numpy()

    overall = compute_summary_stats(confidences)
    percentiles = compute_percentiles(confidences)
    kde_ok = kde_is_appropriate(confidences)

    tp_stats: dict[str, float] | None = None
    fp_stats: dict[str, float] | None = None
    tp_confidences = np.array([])
    fp_confidences = np.array([])
    gt_analysis = False

    if not args.skip_gt:
        tp_confidences, fp_confidences, gt_available = classify_tp_fp(
            detections_df,
            args.annotations_root,
            args.iou_threshold,
        )
        if gt_available:
            gt_analysis = True
            tp_stats = compute_summary_stats(tp_confidences)
            fp_stats = compute_summary_stats(fp_confidences)

    histogram_path = args.output_dir / "confidence_histogram.png"
    boxplot_path = args.output_dir / "confidence_boxplot.png"
    statistics_path = args.output_dir / "confidence_statistics.csv"
    summary_path = args.output_dir / "confidence_summary.json"

    plot_histogram_and_kde(confidences, histogram_path, kde_ok)
    plot_boxplot(confidences, boxplot_path)
    save_statistics_csv(statistics_path, overall, percentiles, tp_stats, fp_stats)

    summary: dict[str, Any] = {
        "source": source,
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "overall": overall,
        "percentiles": percentiles,
        "kde_plotted": kde_ok,
        "outputs": {
            "histogram": str(histogram_path),
            "boxplot": str(boxplot_path),
            "statistics_csv": str(statistics_path),
            "summary_json": str(summary_path),
        },
    }

    if gt_analysis:
        comparison_path = args.output_dir / "confidence_tp_fp_comparison.png"
        plot_tp_fp_comparison(tp_confidences, fp_confidences, comparison_path)
        summary["gt_matching"] = {
            "available": True,
            "true_positive": tp_stats,
            "false_positive": fp_stats,
            "comparison_figure": str(comparison_path),
        }
    else:
        summary["gt_matching"] = {"available": False}

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print_summary(source, overall, percentiles, tp_stats, fp_stats)

    print()
    print("Saved outputs")
    print(f"  {histogram_path}")
    print(f"  {boxplot_path}")
    print(f"  {statistics_path}")
    print(f"  {summary_path}")
    if gt_analysis:
        print(f"  {summary['gt_matching']['comparison_figure']}")
    print()


if __name__ == "__main__":
    main()
