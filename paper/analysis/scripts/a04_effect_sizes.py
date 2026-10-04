#!/usr/bin/env python3
"""Analysis 4 — point-estimate deltas between conditions, per model.

Delta = metric(to) - metric(from), using master_results.csv. No significance is
assessed here; image-level bootstrap CIs for the same deltas are in a05.
Covers every model with five usable conditions. For a pair involving a partial condition the
two marginal metrics are over different valid sets; n_from / n_to record both and
same_n flags it. LaTeX tables cover the seven-model panel (all full_clean).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DELTA_METRICS = {
    "align_accuracy": "dAcc", "align_balanced_accuracy": "dBalAcc", "align_sensitivity": "dSens",
    "align_specificity": "dSpec", "align_f1": "dF1", "uncertain_rate": "dU", "reliable_rate": "dR",
    "unreliable_rate": "dUr", "coverage": "dCov",
}


def main() -> None:
    C.ensure_dirs()
    master = pd.read_csv(C.OUT / "master_results.csv")
    df = C.load_samples(include_partial=True)
    models = C.complete_models(df)
    rows = []
    for m in models:
        g = master[master["model"] == m].set_index("condition")
        for group, pairs in C.PAIR_GROUPS.items():
            for a, b in pairs:
                row = {"model": m, "group": group, "pair": f"{a}->{b}",
                       "n_from": int(g.loc[a, "n_samples"]), "n_to": int(g.loc[b, "n_samples"]),
                       "same_n": bool(g.loc[a, "n_samples"] == g.loc[b, "n_samples"])}
                for col, name in DELTA_METRICS.items():
                    row[name] = g.loc[b, col] - g.loc[a, col]
                rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(C.OUT / "condition_deltas.csv", index=False)

    def table(pairs, path, label, caption):
        out = []
        for m in [x for x in models if x in C.CORE_PANEL]:
            for p in pairs:
                s = d[(d["model"] == m) & (d["pair"] == p)].iloc[0]
                out.append({"Model": m, "Pair": p.replace("->", r"$\rightarrow$"),
                            **{k: f"{s[k]:+.3f}" for k in ("dAcc", "dBalAcc", "dSens", "dSpec", "dF1",
                                                            "dR", "dU", "dUr")}})
        t = pd.DataFrame(out).rename(columns={
            "dAcc": r"$\Delta$Acc", "dBalAcc": r"$\Delta$BalAcc", "dSens": r"$\Delta$Sens",
            "dSpec": r"$\Delta$Spec", "dF1": r"$\Delta$F1", "dR": r"$\Delta$R", "dU": r"$\Delta$U",
            "dUr": r"$\Delta$Ur"})
        C.write_latex_table(path, t, caption=caption, label=label, colspec="llrrrrrrrr",
                            group_col="Model", escape_cells=False)

    table(["A2->A4", "A2->A5"], C.TABLES / "tab_deltas_image_representation.tex", "tab:deltas_image",
          r"Point-estimate changes from A2 to A4 and A5 (to $-$ from). Accuracy, BalAcc, Sens, Spec and F1 "
          r"are Protocol-v2 IoU-alignment metrics on decided samples; R/U/Ur are rates over $N$. The "
          r"confidence text is identical; the image and the image-description paragraph differ. No "
          r"significance is implied; see the bootstrap intervals.")
    table(["A1->A2", "A2->A3"], C.TABLES / "tab_deltas_text_only.tex", "tab:deltas_text",
          r"Point-estimate changes for the text-only contrasts (identical image). Same conventions as "
          r"Table~\ref{tab:deltas_image}.")

    # Figure: deltas for A2->A4 and A2->A5, selected metrics
    show = ["dBalAcc", "dSens", "dSpec", "dU", "dR"]
    fig, axes = plt.subplots(1, len(show), figsize=(13, 3.2), sharey=True)
    for ax, k in zip(axes, show):
        for j, (p, mk) in enumerate((("A2->A4", "o"), ("A2->A5", "s"))):
            sub = d[d["pair"] == p].set_index("model").reindex(models)
            ax.scatter(sub[k], [i + (j - 0.5) * 0.25 for i in range(len(models))], marker=mk, s=18,
                       label=p.replace("->", "→"))
        ax.axvline(0, color="0.5", lw=0.7)
        ax.set_title(k.replace("d", "Δ", 1), fontsize=9)
        ax.set_yticks(range(len(models)), models, fontsize=7)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=7, frameon=False)
    fig.text(0.01, 0.01, "Alignment metrics (BalAcc, Sens, Spec) are Protocol-v2 IoU-alignment quantities. "
             "Point estimates; CIs in fig_bootstrap_deltas.", fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_deltas_image_representation.{ext}", dpi=200)
    plt.close(fig)
    print(d[d["pair"].isin(["A2->A4", "A2->A5"])].round(3).to_string())


if __name__ == "__main__":
    main()
