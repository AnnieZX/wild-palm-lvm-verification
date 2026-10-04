#!/usr/bin/env python3
"""Analysis 3 — per-detection decision transitions between conditions.

Pairs are grouped by what the executed inputs change (see common.PAIR_GROUPS):
  text_only                A1->A2, A2->A3, A1->A3   identical image file
  image_representation     A2->A4, A2->A5, A4->A5   identical confidence text; image and the
                                                    image-description paragraph change
  mixed_image_and_metadata A1->A4, A1->A5, A3->A4, A3->A5
Models with all five conditions usable (full_clean or partial) are used. Each pair uses the
detections with a valid decision in BOTH conditions; n is reported per pair, so a partial
condition never silently changes the denominator. Direction uses the order
Reliable < Uncertain < Unreliable; "toward rejection"/"toward acceptance" name the
direction of a shift and carry no value judgement.
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

ORDER = {lab: i for i, lab in enumerate(C.LABELS)}


def main() -> None:
    C.ensure_dirs()
    df = C.load_samples(include_partial=True)
    models = C.complete_models(df)
    status = C.cell_status(df)
    wide = df[df["display"].isin(models)].pivot_table(
        index=["display", "sample_id"], columns="condition", values="decision", aggfunc="first")

    mat_rows, sum_rows = [], []
    for m in models:
        w_all = wide.loc[m]
        for group, pairs in C.PAIR_GROUPS.items():
            for a, b in pairs:
                w = C.paired(w_all, a, b)
                ct = pd.crosstab(w[a], w[b]).reindex(index=list(C.LABELS), columns=list(C.LABELS), fill_value=0)
                n = int(ct.values.sum())
                for fr in C.LABELS:
                    row_total = int(ct.loc[fr].sum())
                    for to in C.LABELS:
                        k = int(ct.loc[fr, to])
                        mat_rows.append({"model": m, "group": group, "from_condition": a, "to_condition": b,
                                         "from_label": fr, "to_label": to, "count": k,
                                         "share_of_n": k / n,
                                         "share_of_from_row": k / row_total if row_total else np.nan})
                fr_idx = w[a].map(ORDER).to_numpy()
                to_idx = w[b].map(ORDER).to_numpy()
                changed = fr_idx != to_idx
                rej = to_idx > fr_idx
                acc = to_idx < fr_idx
                sum_rows.append({
                    "model": m, "group": group, "pair": f"{a}->{b}", "n": n,
                    "n_excluded_invalid": C.N_COHORT - n,
                    "pair_status": "full_clean" if status[(m, a)] == status[(m, b)] == "full_clean" else "partial",
                    "n_changed": int(changed.sum()), "changed_rate": changed.mean(),
                    "toward_rejection_rate": rej.mean(), "toward_acceptance_rate": acc.mean(),
                    "net_toward_rejection_rate": rej.mean() - acc.mean(),
                    **{f"{C.LABEL_SHORT[fr]}->{C.LABEL_SHORT[to]}": int(ct.loc[fr, to])
                       for fr in C.LABELS for to in C.LABELS if fr != to},
                })
    mats = pd.DataFrame(mat_rows)
    summ = pd.DataFrame(sum_rows)
    mats.to_csv(C.OUT / "transition_matrices_long.csv", index=False)
    summ.to_csv(C.OUT / "transition_summary.csv", index=False)

    # LaTeX: changed % and net shift for the five primary pairs
    primary = [("text_only", "A1->A2"), ("text_only", "A2->A3"),
               ("image_representation", "A2->A4"), ("image_representation", "A2->A5"),
               ("image_representation", "A4->A5")]
    def rows_for(ms, mark_partial):
        rows = []
        for m in ms:
            row = {"Model": m}
            for g, p in primary + [("mixed_image_and_metadata", "A1->A5")]:
                s = summ[(summ["model"] == m) & (summ["pair"] == p)].iloc[0]
                row[p] = f"{100*s['changed_rate']:.1f} ({100*s['net_toward_rejection_rate']:+.1f})"
                if mark_partial and s["pair_status"] == "partial":
                    row[p] += rf"$^\dagger$"
            rows.append(row)
        return pd.DataFrame(rows)
    core = [m for m in models if m in C.CORE_PANEL]
    ext = [m for m in models if m not in C.CORE_PANEL]
    C.write_latex_table(
        C.TABLES / "tab_transitions_summary_extended.tex", rows_for(ext, True),
        caption=r"As Table~\ref{tab:transitions_summary}, for the further checkpoints with all five conditions. "
                r"$^\dagger$ pair involves a partial condition: computed over detections valid in both "
                r"conditions (n in transition\_summary.csv). A checkpoint that answers a single label in both "
                r"conditions shows 0.0 by construction (see the label-usage table).",
        label="tab:transitions_summary_extended", colspec="lcccccc", escape_cells=False)
    C.write_latex_table(
        C.TABLES / "tab_transitions_summary.tex", rows_for(core, False),
        caption=r"Share of detections whose decision changes between two conditions, \% of $N=5{,}747$ "
                r"(in parentheses: net shift toward rejection, \% of $N$; negative = net shift toward "
                r"acceptance). A1$\rightarrow$A2 and A2$\rightarrow$A3 change only prompt text over an "
                r"identical image. A2$\rightarrow$A4, A2$\rightarrow$A5 and A4$\rightarrow$A5 keep the "
                r"confidence text fixed but change the image \emph{and} the matching image-description "
                r"paragraph, so they are image-representation contrasts, not image-only contrasts.",
        label="tab:transitions_summary", colspec="lcccccc", escape_cells=False)

    # Heatmaps: one figure per group, rows = models, cols = pairs; row-normalised shares
    for group, pairs in C.PAIR_GROUPS.items():
        fig, axes = plt.subplots(len(models), len(pairs), figsize=(2.3 * len(pairs), 2.0 * len(models)),
                                 squeeze=False)
        for i, m in enumerate(models):
            for j, (a, b) in enumerate(pairs):
                ax = axes[i, j]
                sub = mats[(mats["model"] == m) & (mats["from_condition"] == a) & (mats["to_condition"] == b)]
                share = sub.pivot(index="from_label", columns="to_label", values="share_of_from_row") \
                    .reindex(index=list(C.LABELS), columns=list(C.LABELS))
                cnt = sub.pivot(index="from_label", columns="to_label", values="count") \
                    .reindex(index=list(C.LABELS), columns=list(C.LABELS))
                ax.imshow(share.fillna(0).to_numpy(), vmin=0, vmax=1, cmap="Blues")
                for r in range(3):
                    for c in range(3):
                        k = int(cnt.iloc[r, c])
                        sh = share.iloc[r, c]
                        txt = f"{k}" if np.isnan(sh) else f"{k}\n{100*sh:.0f}%"
                        ax.text(c, r, txt, ha="center", va="center", fontsize=5.5,
                                color="white" if (not np.isnan(sh) and sh > 0.6) else "black")
                ax.set_xticks(range(3), ["R", "U", "Ur"], fontsize=6)
                ax.set_yticks(range(3), ["R", "U", "Ur"], fontsize=6)
                if i == 0:
                    ax.set_title(f"{a} → {b}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{m}\nfrom {a}", fontsize=7)
                ax.set_xlabel(f"to {b}", fontsize=6)
        fig.suptitle(f"Decision transitions — {group.replace('_', ' ')}\n{C.PAIR_GROUP_NOTES[group]}",
                     fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "pdf"):
            fig.savefig(C.FIGURES / f"fig_transitions_{group}.{ext}", dpi=200)
        plt.close(fig)
    print(summ[["model", "pair", "changed_rate", "net_toward_rejection_rate"]].round(3).to_string())


if __name__ == "__main__":
    main()
