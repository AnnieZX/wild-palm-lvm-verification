#!/usr/bin/env python3
"""
Purpose:
    Figures for the ground-truth-independent detection-property behavior analysis.
    Reads only event_rates_by_bin.csv and analysis_info.json written by
    scripts/analysis/detection_property_behavior_analysis.py. Shaded bands are 95%
    image-clustered bootstrap intervals. Figures describe decision behavior only.

Input (default outputs/detection_property_behavior_analysis/):
    - event_rates_by_bin.csv, analysis_info.json

Output (default outputs/detection_property_behavior_analysis/figures/):
    - change_rate_by_confidence.png
    - change_rate_by_box_area.png
    - into_uncertain_qwen3_internvl.png
    - into_unreliable_qwen2_5.png
    - glm_gemma_phi_transitions.png

Example:
    python scripts/visualization/plot_detection_property_behavior.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from detection_property_behavior_analysis import DEFAULT_OUT_DIR, MODELS  # noqa: E402

MODEL_COLORS = dict(zip(MODELS, plt.get_cmap("tab10").colors))
PROPERTY_AXIS = {
    "confidence": "YOLO confidence bin",
    "norm_area": "Box area quintile (% of patch area)",
}
PROPERTY_TITLE = {"confidence": "YOLO confidence", "norm_area": "box area (share of patch)"}


def area_tick_labels(info: dict) -> list[str]:
    edges = info["quintile_edges"]["norm_area"]
    return [f"Q{i + 1}\n{100 * a:.1f}–{100 * b:.1f}%" for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def tick_labels(prop: str, rows: pd.DataFrame, info: dict) -> list[str]:
    if prop == "norm_area":
        return area_tick_labels(info)
    return list(rows.sort_values("bin_order")["bin"].drop_duplicates())


def draw(ax, rows: pd.DataFrame, color, label: str, linestyle: str = "-") -> None:
    rows = rows.sort_values("bin_order")
    x = rows["bin_order"].to_numpy()
    ax.plot(x, 100 * rows["rate"], marker="o", color=color, label=label, linestyle=linestyle)
    ax.fill_between(x, 100 * rows["ci_low"], 100 * rows["ci_high"], color=color, alpha=0.15, linewidth=0)


def change_rate_figure(bins: pd.DataFrame, prop: str, info: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, comparison in zip(axes, ("A1->A5", "A2->A5")):
        sub = bins[(bins["comparison"] == comparison) & (bins["event"] == "changed") & (bins["property"] == prop)]
        for key, (name, _) in MODELS.items():
            draw(ax, sub[sub["model_key"] == key], MODEL_COLORS[key], name)
        labels = tick_labels(prop, sub, info)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel(PROPERTY_AXIS[prop])
        ax.set_title(f"{comparison.replace('->', ' → ')}: decision changed", fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Detections with a changed decision (%)")
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"Paired decision change by {PROPERTY_TITLE[prop]} "
                 "(bands: 95% image-clustered bootstrap)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def event_grid(bins: pd.DataFrame, series: list[tuple[str, str, str, str]], info: dict, title: str,
               path: Path, comparison: str = "A1->A5") -> None:
    """series: (model_key, event, legend label, linestyle); one column per property."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, prop in zip(axes, ("norm_area", "confidence")):
        sub = bins[(bins["comparison"] == comparison) & (bins["property"] == prop)]
        for key, event, label, style in series:
            rows = sub[(sub["model_key"] == key) & (sub["event"] == event)]
            draw(ax, rows, MODEL_COLORS[key], label, style)
        labels = tick_labels(prop, sub, info)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel(PROPERTY_AXIS[prop])
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Transition rate among detections at risk (%)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    bins = pd.read_csv(args.analysis_dir / "event_rates_by_bin.csv")
    info = json.loads((args.analysis_dir / "analysis_info.json").read_text(encoding="utf-8"))
    out = args.analysis_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    change_rate_figure(bins, "confidence", info, out / "change_rate_by_confidence.png")
    change_rate_figure(bins, "norm_area", info, out / "change_rate_by_box_area.png")
    event_grid(bins, [
        ("qwen3_vl", "R->U", "Qwen3-VL: A1 Reliable → A5 Uncertain", "-"),
        ("qwen3_vl", "into_U", "Qwen3-VL: into Uncertain (from R or Ur)", "--"),
        ("internvl3_5_hf", "R->U", "InternVL3.5: A1 Reliable → A5 Uncertain", "-"),
        ("internvl3_5_hf", "into_U", "InternVL3.5: into Uncertain (from R or Ur)", "--"),
    ], info, "A1 → A5 transitions into Uncertain (bands: 95% image-clustered bootstrap)",
        out / "into_uncertain_qwen3_internvl.png")
    event_grid(bins, [
        ("qwen2_5_vl", "into_Ur", "Qwen2.5-VL: into Unreliable (from R or U)", "-"),
        ("qwen2_5_vl", "R->Ur", "Qwen2.5-VL: A1 Reliable → A5 Unreliable", "--"),
        ("qwen2_5_vl", "U->Ur", "Qwen2.5-VL: A1 Uncertain → A5 Unreliable", ":"),
    ], info, "Qwen2.5-VL A1 → A5 transitions into Unreliable (bands: 95% image-clustered bootstrap)",
        out / "into_unreliable_qwen2_5.png")
    event_grid(bins, [
        ("glm_4_6v_flash", "into_U", "GLM-4.6V: into Uncertain", "-"),
        ("glm_4_6v_flash", "into_Ur", "GLM-4.6V: into Unreliable", "--"),
        ("gemma4", "R->Ur", "Gemma-4: Reliable → Unreliable", "-"),
        ("phi4_multimodal", "R->Ur", "Phi-4: Reliable → Unreliable", "-"),
    ], info, "A1 → A5: GLM-4.6V abstention vs rejection; Gemma-4 / Phi-4 Reliable → Unreliable flips",
        out / "glm_gemma_phi_transitions.png")
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
