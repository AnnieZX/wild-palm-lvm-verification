#!/usr/bin/env python3
"""Analysis 6 — cross-model agreement on identical sample IDs, separately per condition.

Per condition and model pair (3-class R/U/Ur):
  raw agreement p_o, chance agreement p_e (product of marginals), Cohen's kappa, and
  kappa_max = (sum_k min(p_a,k, p_b,k) - p_e) / (1 - p_e), the largest kappa attainable
  given the two models' marginals. kappa_max < 1 flags marginal imbalance (e.g. a model
  that never answers Uncertain) that caps kappa regardless of item-level agreement.
  Secondary: kappa on the binary split Reliable vs not-Reliable.
Per condition (all models with that condition):
  Fleiss' kappa (assumes interchangeable raters; models are fixed, distinct raters, so it is
  reported only as a summary) and Light's kappa (mean pairwise Cohen's kappa, the
  fixed-rater alternative).
Only full_clean cells are used (identical N = 5,747 for every rater). The LaTeX summary and
agreement_by_condition_complete_models.csv use the seven-model panel (common.CORE_PANEL);
agreement_by_condition_all_clean_complete.csv uses every model with five full_clean conditions.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def kappa_max(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.array([np.mean(a == l) for l in C.LABELS])
    pb = np.array([np.mean(b == l) for l in C.LABELS])
    pe = float(np.sum(pa * pb))
    pomax = float(np.sum(np.minimum(pa, pb)))
    return (pomax - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    C.ensure_dirs()
    df = C.load_samples()
    wide = df.pivot_table(index="sample_id", columns=["condition", "display"], values="decision", aggfunc="first")

    pair_rows, cond_rows = [], []
    for cond in C.CONDITIONS:
        models = [m for m in C.MODEL_ORDER if (cond, m) in wide.columns]
        w = wide[cond][models]
        assert not w.isna().any().any(), "missing decisions on identical sample IDs"
        for a, b in itertools.combinations(models, 2):
            xa, xb = w[a].to_numpy(), w[b].to_numpy()
            po, pe, k = C.cohen_kappa(xa, xb)
            _, _, k_bin = C.cohen_kappa(np.where(xa == "Reliable", "R", "notR"),
                                        np.where(xb == "Reliable", "R", "notR"), labels=("R", "notR"))
            pair_rows.append({"condition": cond, "model_a": a, "model_b": b, "n": len(xa),
                              "raw_agreement": po, "chance_agreement": pe, "cohen_kappa": k,
                              "kappa_max_given_marginals": kappa_max(xa, xb),
                              "cohen_kappa_reliable_vs_not": k_bin})
        counts = np.stack([(w == lab).sum(axis=1).to_numpy() for lab in C.LABELS], axis=1)
        sub = [r for r in pair_rows if r["condition"] == cond]
        cond_rows.append({"condition": cond, "n_models": len(models), "models": ";".join(models), "n": len(w),
                          "fleiss_kappa": C.fleiss_kappa(counts),
                          "light_kappa_mean_pairwise": float(np.mean([r["cohen_kappa"] for r in sub])),
                          "mean_pairwise_raw_agreement": float(np.mean([r["raw_agreement"] for r in sub])),
                          "share_items_unanimous": float(np.mean(counts.max(axis=1) == len(models)))})
        # same summary restricted to the 7 complete models for A1 (comparable model set across conditions)
    pairs = pd.DataFrame(pair_rows)
    conds = pd.DataFrame(cond_rows)

    def panel_summary(complete):
        out = []
        for cond in C.CONDITIONS:
            w = wide[cond][complete]
            counts = np.stack([(w == lab).sum(axis=1).to_numpy() for lab in C.LABELS], axis=1)
            sub = pairs[(pairs["condition"] == cond) & pairs["model_a"].isin(complete) & pairs["model_b"].isin(complete)]
            out.append({"condition": cond, "n_models": len(complete), "fleiss_kappa": C.fleiss_kappa(counts),
                        "light_kappa_mean_pairwise": sub["cohen_kappa"].mean(),
                        "mean_pairwise_raw_agreement": sub["raw_agreement"].mean(),
                        "share_items_unanimous": float(np.mean(counts.max(axis=1) == len(complete)))})
        return pd.DataFrame(out)
    conds7 = panel_summary([m for m in C.MODEL_ORDER if m in C.CORE_PANEL])
    all_clean = C.complete_models(df)
    conds_all = panel_summary(all_clean).assign(models=";".join(all_clean))

    pairs.to_csv(C.OUT / "agreement_pairwise.csv", index=False)
    conds.to_csv(C.OUT / "agreement_by_condition_all_models.csv", index=False)
    conds7.to_csv(C.OUT / "agreement_by_condition_complete_models.csv", index=False)
    conds_all.to_csv(C.OUT / "agreement_by_condition_all_clean_complete.csv", index=False)

    t = conds7.copy()
    t = pd.DataFrame({"Cond.": t["condition"], "Models": t["n_models"].map(str),
                      "Mean raw agr.": t["mean_pairwise_raw_agreement"].map(C.fmt),
                      r"Light's $\kappa$": t["light_kappa_mean_pairwise"].map(C.fmt),
                      r"Fleiss' $\kappa$": t["fleiss_kappa"].map(C.fmt),
                      "Unanimous": t["share_items_unanimous"].map(C.fmt)})
    C.write_latex_table(
        C.TABLES / "tab_agreement_by_condition.tex", t,
        caption=r"Agreement among the seven models with all five conditions, on identical sample IDs (3-class). "
                r"Light's $\kappa$ is the mean pairwise Cohen's $\kappa$. Fleiss' $\kappa$ assumes "
                r"interchangeable raters and is shown only as a summary. Several models never answer "
                r"Uncertain, which caps attainable $\kappa$ (see pairwise $\kappa_{\max}$ in the CSV).",
        label="tab:agreement_by_condition", colspec="lrrrrr", escape_cells=False)

    # Heatmaps: raw agreement (upper triangle) and kappa (lower triangle) per condition
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    for ax, cond in zip(axes, C.CONDITIONS):
        models = [m for m in C.MODEL_ORDER if (cond, m) in wide.columns]
        n = len(models)
        mat = np.full((n, n), np.nan)
        sub = pairs[pairs["condition"] == cond]
        for r in sub.itertuples():
            i, j = models.index(r.model_a), models.index(r.model_b)
            mat[i, j] = r.raw_agreement
            mat[j, i] = r.cohen_kappa
        ax.imshow(np.nan_to_num(mat, nan=1.0), vmin=-0.2, vmax=1, cmap="viridis")
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if mat[i, j] < 0.55 else "black")
        ax.set_xticks(range(n), models, rotation=90, fontsize=6)
        ax.set_yticks(range(n), models, fontsize=6)
        ax.set_title(f"{cond}  (upper: raw agreement, lower: Cohen's κ)", fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_agreement_heatmaps.{ext}", dpi=200)
    plt.close(fig)
    print(conds.round(3).to_string())
    print(conds7.round(3).to_string())
    print(conds_all.drop(columns="models").round(3).to_string())


if __name__ == "__main__":
    main()
