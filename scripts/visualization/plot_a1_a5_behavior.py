#!/usr/bin/env python3
"""
Purpose:
    Figures for the ground-truth-independent A1-A5 behavior analysis. Reads only the
    CSVs written by scripts/analysis/a1_a5_behavior_analysis.py; computes nothing new
    beyond row shares for display. Figures describe decision behavior only and carry
    no correctness terminology.

Input (default outputs/a1_a5_behavior_analysis/):
    - table_b_decision_distribution.csv
    - table_c_transitions_a1_reference.csv

Output (default outputs/a1_a5_behavior_analysis/figures/):
    - decision_distribution_by_condition.png
    - transition_heatmaps_A1_to_A5.png
    - decision_change_from_A1.png

Example:
    python scripts/visualization/plot_a1_a5_behavior.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from a1_a5_behavior_analysis import CONDITIONS, DEFAULT_OUT_DIR, LABELS, cell_name  # noqa: E402

LABEL_COLORS = {"Reliable": "#4C72B0", "Uncertain": "#DDAA33", "Unreliable": "#C44E52"}


def plot_distributions(table_b: pd.DataFrame, path: Path) -> None:
    models = list(dict.fromkeys(table_b["model"]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for ax, model in zip(axes.flat, models):
        rows = table_b[table_b["model"] == model].set_index("condition").loc[list(CONDITIONS)]
        bottom = np.zeros(len(rows))
        for label in LABELS:
            values = rows[f"{label.lower()}_pct"].to_numpy()
            ax.bar(rows.index, values, bottom=bottom, color=LABEL_COLORS[label], label=label, width=0.7)
            bottom += values
        ax.set_title(model, fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of 5,747 detections (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("VLM decision distribution across input conditions A1–A5", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_transition_heatmaps(table_c: pd.DataFrame, path: Path, comparison: str = "A1->A5") -> None:
    rows = table_c[table_c["comparison"] == comparison]
    src, dst = comparison.split("->")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        counts = np.array([[row[cell_name(a, b)] for b in LABELS] for a in LABELS], dtype=float)
        totals = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                text = f"{int(counts[i, j])}\n({100 * shares[i, j]:.0f}%)" if totals[i, 0] else "—"
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        color="white" if shares[i, j] > 0.6 else "black")
        ax.set_xticks(range(3), LABELS, fontsize=9)
        ax.set_yticks(range(3), [f"{a}\n(n={int(t)})" for a, t in zip(LABELS, totals[:, 0])], fontsize=9)
        ax.set_xlabel(f"{dst} decision")
        ax.set_ylabel(f"{src} decision")
        ax.set_title(f"{row['model']}  (changed {row['changed_pct']:.1f}%)", fontsize=10)
    fig.suptitle(f"Paired per-detection decision transitions {src} → {dst} "
                 "(cell = count; shade and % = share of the row)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_change_rates(table_c: pd.DataFrame, path: Path) -> None:
    models = list(dict.fromkeys(table_c["model"]))
    comparisons = [f"A1->{c}" for c in CONDITIONS[1:]]
    panels = [
        ("changed_pct", "Changed decision (%)"),
        ("toward_rejection_pct", "Shift toward rejection (%)\nR→U, R→X, U→X"),
        ("toward_acceptance_pct", "Shift toward acceptance (%)\nX→U, X→R, U→R"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, (column, title) in zip(axes, panels):
        for k, model in enumerate(models):
            rows = table_c[table_c["model"] == model].set_index("comparison").loc[comparisons]
            ax.plot([c.replace("->", "→") for c in comparisons], rows[column], marker="o",
                    color=cmap(k), label=model)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of 5,747 detections (%)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Paired decision change relative to A1 (behavioral shift only; no direction is preferable)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    table_b = pd.read_csv(args.analysis_dir / "table_b_decision_distribution.csv")
    table_c = pd.read_csv(args.analysis_dir / "table_c_transitions_a1_reference.csv")
    out = args.analysis_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_distributions(table_b, out / "decision_distribution_by_condition.png")
    plot_transition_heatmaps(table_c, out / "transition_heatmaps_A1_to_A5.png")
    plot_change_rates(table_c, out / "decision_change_from_A1.png")
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
