#!/usr/bin/env python3
"""Analysis 2 — decision distributions (Reliable / Uncertain / Unreliable rates).

Purely descriptive; no collapse or abstention threshold is applied (label-usage and
collapse diagnostics are in a10). Figures shade A1-A3 to mark that these three conditions
share an identical image file. Includes partial cells: rates are over the valid rows (n),
and condition_status marks them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COLORS = {"Reliable": "#4c78a8", "Uncertain": "#bab0ac", "Unreliable": "#e45756"}


def main() -> None:
    C.ensure_dirs()
    df = C.load_samples(include_partial=True)
    status = C.cell_status(df)
    rows = []
    for (key, disp, cond), sub in df.groupby(["model_key", "display", "condition"], sort=False):
        n = len(sub)
        for lab in C.LABELS:
            k = int((sub["decision"] == lab).sum())
            rows.append({"model_key": key, "model": disp, "condition": cond, "label": lab,
                         "count": k, "n": n, "rate": k / n, "condition_status": status[(disp, cond)]})
    long = pd.DataFrame(rows)
    long["model"] = pd.Categorical(long["model"], C.MODEL_ORDER, ordered=True)
    long = long.sort_values(["model", "condition", "label"])
    long.to_csv(C.OUT / "decision_distribution_long.csv", index=False)

    wide = long.pivot_table(index=["model", "condition"], columns="label", values="rate", observed=True)
    wide = wide.reset_index()
    meta = long.drop_duplicates(["model", "condition"]).set_index(["model", "condition"])[["n", "condition_status"]]
    wide = wide.join(meta, on=["model", "condition"])
    wide["uses_uncertain"] = wide["Uncertain"] > 0
    wide["max_class_share"] = wide[list(C.LABELS)].max(axis=1)
    wide["max_class"] = wide[list(C.LABELS)].idxmax(axis=1)
    wide.to_csv(C.OUT / "decision_distribution_wide.csv", index=False)

    # per-model summary across conditions (descriptive ranges, no thresholds)
    summ = []
    for m, g in wide.groupby("model", observed=True):
        summ.append({"model": m, "n_conditions": len(g),
                     "n_partial_conditions": int((g["condition_status"] == "partial").sum()),
                     "n_min": int(g["n"].min()),
                     "reliable_min": g["Reliable"].min(), "reliable_max": g["Reliable"].max(),
                     "uncertain_min": g["Uncertain"].min(), "uncertain_max": g["Uncertain"].max(),
                     "unreliable_min": g["Unreliable"].min(), "unreliable_max": g["Unreliable"].max(),
                     "conditions_with_any_uncertain": int((g["Uncertain"] > 0).sum()),
                     "conditions_with_zero_uncertain": int((g["Uncertain"] == 0).sum())})
    pd.DataFrame(summ).to_csv(C.OUT / "decision_distribution_ranges.csv", index=False)

    # LaTeX: model rows x condition columns, "R / U / Ur" percentages
    tab = []
    for m in C.MODEL_ORDER:
        g = wide[wide["model"] == m].set_index("condition")
        if g.empty:
            continue
        row = {"Model": m}
        for c in C.CONDITIONS:
            if c in g.index:
                r = g.loc[c]
                row[c] = f"{100*r['Reliable']:.1f} / {100*r['Uncertain']:.1f} / {100*r['Unreliable']:.1f}"
                if r["condition_status"] == "partial":
                    row[c] += rf"$^\dagger$ ({int(r['n']):,})".replace(",", "{,}")
            else:
                row[c] = "not run"
        tab.append(row)
    C.write_latex_table(
        C.TABLES / "tab_decision_distribution.tex", pd.DataFrame(tab),
        caption=r"Decision distribution, \% Reliable / Uncertain / Unreliable of $N=5{,}747$, per model and "
                r"condition. $^\dagger$ partial condition: share of the valid decisions (count in parentheses); "
                r"parse-error rows are excluded, not repaired. A1--A3 share an identical image and differ only in "
                r"prompt metadata text. No collapse or abstention threshold is applied.",
        label="tab:decision_distribution_generated", colspec="lccccc", escape_cells=False)

    # Figure 1: stacked bars, small multiples
    models = [m for m in C.MODEL_ORDER if m in set(wide["model"])]
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.6 * nrow), sharey=True)
    for ax, m in zip(axes.flat, models):
        g = wide[wide["model"] == m].set_index("condition").reindex(C.CONDITIONS)
        x = np.arange(len(C.CONDITIONS))
        bottom = np.zeros(len(x))
        for lab in C.LABELS:
            vals = g[lab].fillna(0).to_numpy()
            ax.bar(x, vals, bottom=bottom, color=COLORS[lab], width=0.75, label=lab)
            bottom += vals
        for i, c in enumerate(C.CONDITIONS):
            if np.isnan(g.loc[c, "Reliable"]):
                ax.text(i, 0.5, "not run", ha="center", va="center", rotation=90, fontsize=7, color="0.4")
        ax.axvspan(-0.5, 2.5, color="0.92", zorder=-1)
        ax.set_xticks(x, C.CONDITIONS)
        ax.set_title(m, fontsize=9)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.text(0.5, 0.005, "Shaded: A1–A3 use the identical image (text-only differences).",
             ha="center", fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_decision_distribution_stacked.{ext}", dpi=200)
    plt.close(fig)

    # Figure 2: one panel per label, rate vs condition, one marker series per model
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    markers = "osD^v<>ph*"
    for ax, lab in zip(axes, C.LABELS):
        for i, m in enumerate(models):
            g = wide[wide["model"] == m].set_index("condition").reindex(C.CONDITIONS)
            ax.plot(range(5), g[lab], marker=markers[i % len(markers)], lw=0.8, ms=4, label=m)
        ax.axvspan(-0.3, 2.3, color="0.92", zorder=-1)
        ax.set_xticks(range(5), C.CONDITIONS)
        ax.set_title(f"{lab} rate", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("share of valid decisions (N = 5,747 unless partial)")
    axes[-1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.text(0.01, 0.005, "Lines connect conditions for readability only; A1–A5 are not an ordered scale. "
             "Shaded: identical image (A1–A3).", fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_decision_rates_by_condition.{ext}", dpi=200)
    plt.close(fig)
    print(pd.DataFrame(summ).round(3).to_string())


if __name__ == "__main__":
    main()
