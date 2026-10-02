"""
Generate the README figures under docs/assets/readme/ (light + dark variants).

Figures:
  ablation_inputs_{light,dark}.png   A1–A5 inputs rendered with the repo's own builders
  decision_mix_{light,dark}.png      R/U/Ur distribution per model × condition @5747 (all 6 complete models)
  specificity_{light,dark}.png       Protocol-v2 alignment specificity (share of decided LabelMe-unmatched
                                     detections answered Unreliable) per model × condition @5747
  a1_behavior_{light,dark}.png       A1 decision mix across every A1-evaluated model

Numbers are transcribed from docs/EXPERIMENT_RESULTS_CANONICAL.md (Evaluation
Protocol v2; source artifacts outputs/evaluation_protocol_v2/). No inference is run.
R/U/Ur counts are prediction-only and identical under Protocols v1 and v2.

Usage:
    python scripts/visualization/make_readme_figures.py [--only specificity a1_behavior]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.ablation_verification_images import (  # noqa: E402
    build_a4_combined_image,
    build_a5_crop_only_image,
)
from src.preprocessing.gt_palm_bboxes import is_palm_label  # noqa: E402
from src.preprocessing.verification_overlay import render_single_detection_overlay  # noqa: E402

OUT = ROOT / "docs" / "assets" / "readme"
# TODO(DEAC sync): replace the ablation-gallery target with a REAL YOLO candidate.
# No YOLO detection is tracked in this repository, so the gallery currently uses a
# LabelMe `palm` annotation as a stand-in box (the README caption says so). After
# syncing a real example from DEAC (e.g. a row of a verification_dataset
# prompt_index.csv plus its raw patch), set YOLO_CANDIDATE to a dict like
#     {"image": ROOT / "data/samples/images/<patch>.png",
#      "bbox_xywh": (x, y, w, h)}          # exact YOLO box, pixel coordinates
# then re-run this script and drop the "Illustration only" note from the README.
YOLO_CANDIDATE: dict | None = None

# Fallback stand-in (LabelMe annotation, NOT a YOLO detection).
SAMPLE_IMAGE = ROOT / "data" / "samples" / "images" / "100_0003_0001_2.png"
SAMPLE_JSON = ROOT / "data" / "samples" / "json" / "100_0003_0001_2.json"
SAMPLE_PALM_INDEX = 1

CONDITIONS = ["A1", "A2", "A3", "A4", "A5"]
# Every model with a completed full A1–A5 @5747 run: four useful verifiers, then
# InternVL3.5 (abstention-heavy, condition-dependent), then Gemma 4 (Reliable-heavy collapse).
MODELS = ["Qwen2.5-VL", "Qwen3-VL", "GLM-4.6V-Flash", "Phi-4 MM", "InternVL3.5-8B", "Gemma 4 12B"]
N_USEFUL = 4
COLLAPSED = {"Gemma 4 12B"}
OUTCOME = {
    "Qwen2.5-VL": "useful verifier",
    "Qwen3-VL": "useful verifier",
    "GLM-4.6V-Flash": "useful verifier",
    "Phi-4 MM": "useful verifier",
    "InternVL3.5-8B": "abstention-heavy, condition-dependent",
    "Gemma 4 12B": "Reliable-heavy collapse",
}

# R / U / Ur @5747 — docs/EXPERIMENT_RESULTS_CANONICAL.md §5
DECISIONS = {
    "Qwen2.5-VL": [(3593, 1775, 379), (4268, 1344, 135), (4416, 1313, 18), (3570, 1557, 620), (3065, 455, 2227)],
    "Qwen3-VL": [(4309, 246, 1192), (4636, 401, 710), (4221, 367, 1159), (5051, 276, 420), (1081, 3948, 718)],
    "GLM-4.6V-Flash": [(3719, 50, 1978), (3783, 57, 1907), (3696, 100, 1951), (3960, 64, 1723), (2163, 1082, 2502)],
    "Phi-4 MM": [(4985, 0, 762), (5012, 0, 735), (4132, 0, 1615), (3481, 0, 2266), (4160, 0, 1587)],
    "InternVL3.5-8B": [(3085, 2570, 92), (3497, 2239, 11), (3420, 2298, 29), (3343, 2399, 5), (1515, 4078, 154)],
    "Gemma 4 12B": [(5583, 0, 164), (5595, 0, 152), (5424, 0, 323), (5547, 0, 200), (5261, 0, 486)],
}

# Protocol-v2 alignment specificity @5747 — docs/EXPERIMENT_RESULTS_CANONICAL.md §5
SPECIFICITY = {
    "Qwen2.5-VL": [0.4225, 0.1773, 0.0377, 0.6358, 0.9005],
    "Qwen3-VL": [0.5453, 0.3887, 0.5402, 0.1241, 0.9495],
    "GLM-4.6V-Flash": [0.6566, 0.6785, 0.7349, 0.7005, 0.9062],
    "Phi-4 MM": [0.2571, 0.2774, 0.5141, 0.8574, 0.6959],
    "InternVL3.5-8B": [0.3657, 0.1059, 0.2418, 0.0549, 0.0938],
    "Gemma 4 12B": [0.1034, 0.0909, 0.2320, 0.1850, 0.3323],
}

# A1 decision mix for every model evaluated on A1, Protocol v2 specificity —
# docs/EXPERIMENT_RESULTS_CANONICAL.md §5–§7. Groups: useful verifiers,
# abstention-heavy (condition-dependent), collapse modes.
A1_ALL = [
    ("GLM-4.6V-Flash", "@5747", (3719, 50, 1978), 0.657),
    ("Qwen3-VL", "@5747", (4309, 246, 1192), 0.545),
    ("Qwen2.5-VL", "@5747", (3593, 1775, 379), 0.423),
    ("Phi-4 MM", "@5747", (4985, 0, 762), 0.257),
    ("InternVL3.5-8B", "@5747", (3085, 2570, 92), 0.366),
    ("MiniCPM-V-4.5", "@5747", (5557, 0, 190), 0.113),
    ("Gemma 4 12B", "@5747", (5583, 0, 164), 0.103),
    ("Molmo2-8B", "@5747", (1209, 4537, 1), 0.000),
    ("LLaVA-OneVision", "@1000", (1000, 0, 0), 0.000),
    ("Gemma 3 12B", "@1000", (1000, 0, 0), 0.000),
]
A1_GROUP_BREAKS = [(3.5, "abstention-heavy ↓"), (4.5, "collapse modes ↓")]

# Validated categorical slots (aqua / blue / orange), light and dark steps.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#8a8983",
        "grid": "#e4e3df",
        "series": ["#1baf7a", "#2a78d6", "#eb6834"],
        "seq": ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "frame": "#d6d5d0",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "muted": "#8a8983",
        "grid": "#2e2e2c",
        "series": ["#199e70", "#3987e5", "#d95926"],
        "seq": ["#262624", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
        "frame": "#3a3a37",
    },
}
LABELS = ["Reliable", "Uncertain", "Unreliable"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, name: str, theme: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_{theme}.png"
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def _style_axes(ax, t) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], length=0)


# --------------------------------------------------------------------------- #
# 1. A1–A5 input gallery
# --------------------------------------------------------------------------- #
def _sample_bbox() -> tuple[float, float, float, float]:
    shapes = [s for s in json.loads(SAMPLE_JSON.read_text())["shapes"] if is_palm_label(s.get("label"))]
    pts = shapes[SAMPLE_PALM_INDEX]["points"]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def ablation_inputs(theme: str) -> None:
    t = THEMES[theme]
    if YOLO_CANDIDATE is not None:
        raw = cv2.imread(str(YOLO_CANDIDATE["image"]))
        bbox = tuple(float(v) for v in YOLO_CANDIDATE["bbox_xywh"])
    else:
        raw = cv2.imread(str(SAMPLE_IMAGE))
        bbox = _sample_bbox()
    overlay = render_single_detection_overlay(raw, bbox)
    a4 = build_a4_combined_image(overlay, bbox)
    a5 = build_a5_crop_only_image(raw, bbox)
    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # noqa: E731

    panels = [
        ("A1 · A2 · A3", "Overlay (shared image)", rgb(overlay),
         "A1  no metadata\nA2  + YOLO confidence\nA3  + confidence + bbox geometry"),
        ("A4", "Dual panel: overlay + crop", rgb(a4), "+ YOLO confidence"),
        ("A5", "Crop only, no surround", rgb(a5), "+ YOLO confidence"),
    ]
    fig = plt.figure(figsize=(13, 4.6), facecolor=t["surface"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1], wspace=0.06)
    for i, (tag, title, img, meta) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(t["frame"])
            s.set_linewidth(1)
        ax.set_title(f"{tag}", loc="left", fontsize=13, fontweight="bold", color=t["text"], pad=22)
        ax.text(0, 1.035, title, transform=ax.transAxes, fontsize=10, color=t["text2"], va="bottom")
        ax.text(0, -0.04, meta, transform=ax.transAxes, fontsize=9.5, color=t["text2"],
                va="top", family="DejaVu Sans Mono", linespacing=1.5)
    _save(fig, "ablation_inputs", theme)


# --------------------------------------------------------------------------- #
# 2. Decision mix small multiples
# --------------------------------------------------------------------------- #
def _stacked_row(ax, y, counts, t, height=0.66, label_min=0.07):
    total = sum(counts)
    left = 0.0
    for k, c in enumerate(counts):
        w = c / total
        if w <= 0:
            continue
        ax.barh(y, w, left=left, height=height, color=t["series"][k],
                edgecolor=t["surface"], linewidth=1.5)
        if w >= label_min:
            ax.text(left + w / 2, y, f"{w * 100:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="#ffffff", fontweight="bold")
        left += w


def _legend(fig, t, y=1.0):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in t["series"]]
    fig.legend(handles, LABELS, loc="upper center", bbox_to_anchor=(0.5, y), ncol=3,
               frameon=False, fontsize=10, labelcolor=t["text"], handlelength=1.2)


def decision_mix(theme: str) -> None:
    t = THEMES[theme]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharey=True, facecolor=t["surface"])
    for ax, model in zip(axes.flat, MODELS):
        _style_axes(ax, t)
        collapsed = model in COLLAPSED
        ax.text(1.0, 1.02, OUTCOME[model].replace(", ", ",\n"), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, style="italic", linespacing=1.1,
                color=t["series"][2] if collapsed else t["text2"])
        for i, counts in enumerate(DECISIONS[model]):
            _stacked_row(ax, i, counts, t, label_min=0.1)
        ax.set_xlim(0, 1)
        ax.set_ylim(4.6, -0.6)
        ax.set_yticks(range(5), CONDITIONS)
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(model, loc="left", fontsize=11.5, fontweight="bold", color=t["text"])
        for lbl in ax.get_yticklabels():
            lbl.set_color(t["text"])
            lbl.set_fontweight("bold")
    fig.suptitle("Decision mix per condition  ·  N = 5,747 YOLO detections",
                 x=0.012, y=1.05, ha="left", fontsize=12.5, color=t["text"], fontweight="bold")
    _legend(fig, t, y=1.02)
    fig.subplots_adjust(wspace=0.08, hspace=0.35)
    _save(fig, "decision_mix", theme)


# --------------------------------------------------------------------------- #
# 3. Specificity heatmap
# --------------------------------------------------------------------------- #
def specificity(theme: str) -> None:
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    data = [SPECIFICITY[m] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1.0, aspect="auto")
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            dark_cell = v > 0.5 if theme == "light" else v > 0.65
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color=("#ffffff" if dark_cell else t["text"]) if theme == "light"
                    else ("#0b0b0b" if dark_cell else t["text"]))
    ax.set_xticks(range(5), CONDITIONS)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.tick_params(length=0, colors=t["text"])
    for lbl, model in zip(ax.get_yticklabels(), MODELS):
        if model in COLLAPSED:
            lbl.set_color(t["series"][2])
        elif MODELS.index(model) >= N_USEFUL:
            lbl.set_color(t["text2"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x + 0.5 for x in range(4)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(MODELS) - 1)], minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=3)
    ax.tick_params(which="minor", length=0)
    first_collapsed = min(MODELS.index(m) for m in COLLAPSED)
    for y, label in [(N_USEFUL - 0.5, "abstention-heavy ↓"),
                     (first_collapsed - 0.5, "Reliable-heavy collapse ↓")]:
        ax.axhline(y, color=t["muted"], linewidth=1.2, linestyle=(0, (3, 3)))
        ax.text(1.01, y, label, va="center", ha="left", fontsize=8.5,
                color=t["muted"], style="italic", transform=ax.get_yaxis_transform())
    ax.xaxis.tick_top()
    ax.set_title("Protocol-v2 alignment specificity\n(LabelMe-unmatched classified Unreliable)",
                 loc="left", fontsize=12, fontweight="bold", color=t["text"], pad=30)
    _save(fig, "specificity", theme)


# --------------------------------------------------------------------------- #
# 4. A1 behavior across all models
# --------------------------------------------------------------------------- #
def a1_behavior(theme: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=t["surface"])
    _style_axes(ax, t)
    for i, (_, _, counts, _) in enumerate(A1_ALL):
        _stacked_row(ax, i, counts, t, height=0.7, label_min=0.06)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(A1_ALL) - 0.4, -0.6)
    ax.set_yticks(range(len(A1_ALL)), [f"{m}  {n}" for m, n, _, _ in A1_ALL])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    for lbl in ax.get_yticklabels():
        lbl.set_color(t["text"])
    for i, (_, _, counts, spec) in enumerate(A1_ALL):
        coverage = (counts[0] + counts[2]) / sum(counts)
        note = f"Spec {spec:.2f}" + (f" · cov {coverage:.2f}" if coverage < 0.9 else "")
        ax.text(1.02, i, note, va="center", fontsize=9.5,
                color=t["text2"] if spec >= 0.2 else t["series"][2], fontweight="bold",
                transform=ax.get_yaxis_transform())
    for y, label in A1_GROUP_BREAKS:
        ax.axhline(y, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)))
        ax.text(1.02, y, label, va="center", fontsize=8.5, color=t["muted"],
                transform=ax.get_yaxis_transform(), style="italic",
                bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1))
    fig.suptitle("A1 (overlay only): useful verifiers, abstention-heavy behavior, and collapse modes",
                 x=0.012, y=1.07, ha="left", fontsize=12.5, color=t["text"], fontweight="bold")
    _legend(fig, t, y=1.01)
    _save(fig, "a1_behavior", theme)


FIGURES = {
    "ablation_inputs": ablation_inputs,
    "decision_mix": decision_mix,
    "specificity": specificity,
    "a1_behavior": a1_behavior,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=list(FIGURES),
                        help="Regenerate only these figures (default: all)")
    args = parser.parse_args()
    for theme in THEMES:
        for name in args.only:
            FIGURES[name](theme)


if __name__ == "__main__":
    main()
