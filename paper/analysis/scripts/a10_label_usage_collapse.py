#!/usr/bin/env python3
"""Analysis 10 — label usage and output collapse (ground-truth independent).

Per model x usable condition (full_clean or partial; partial = valid rows only):
  n_valid, n_invalid_excluded, R/U/Ur counts and shares, number of labels used, labels never
  used, majority label and share, Shannon entropy of the 3-class distribution (bits; max
  log2 3 = 1.585) and entropy normalised by log2 3.

Descriptive diagnostics (NOT scientific thresholds; chosen only to make cases easy to find):
  diag_single_label        exactly one label used (exact; no threshold);
  diag_majority_ge_0.99    majority share >= 0.99;
  diag_majority_ge_0.95    majority share >= 0.95 (shown so the 0.99 cut can be checked).
Per checkpoint, a descriptive pattern is derived from these flags across its usable conditions:
  "single label in every usable condition", "majority >= 0.99 in every usable condition",
  "majority >= 0.99 in k of n usable conditions", or "no condition with majority >= 0.99".
No inference about model quality is made; collapse here means concentration of the output
distribution, nothing else.
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

DIAG_THRESHOLDS = (0.99, 0.95)
H_MAX = float(np.log2(3))


def checkpoint_pattern(g: pd.DataFrame) -> str:
    n = len(g)
    if g["diag_single_label"].all():
        return "single label in every usable condition"
    k = int(g["diag_majority_ge_0.99"].sum())
    if k == n:
        return "majority >= 0.99 in every usable condition"
    if k:
        return f"majority >= 0.99 in {k} of {n} usable conditions"
    return "no condition with majority >= 0.99"


def main() -> None:
    C.ensure_dirs()
    out = C.OUT / "label_usage"
    out.mkdir(parents=True, exist_ok=True)
    df = C.load_samples(include_partial=True)
    status = C.cell_status(df)

    rows = []
    for run in C.RUNS:
        for cond in C.CONDITIONS:
            sub = df[(df["model_key"] == run["model_key"]) & (df["condition"] == cond)]
            if sub.empty:
                continue
            counts = np.array([int((sub["decision"] == lab).sum()) for lab in C.LABELS])
            n = int(counts.sum())
            shares = counts / n
            used = [lab for lab, k in zip(C.LABELS, counts) if k > 0]
            maj = int(np.argmax(counts))
            h = C.shannon_entropy(counts) + 0.0  # normalise IEEE -0.0 for single-label cells
            row = {"model_key": run["model_key"], "model": run["display"], "family": run["family"],
                   "condition": cond, "condition_status": status[(run["display"], cond)],
                   "n_valid": n, "n_invalid_excluded": C.N_COHORT - n,
                   **{f"n_{C.LABEL_SHORT[l]}": int(k) for l, k in zip(C.LABELS, counts)},
                   **{f"share_{C.LABEL_SHORT[l]}": float(s) for l, s in zip(C.LABELS, shares)},
                   "labels_used": len(used),
                   "labels_never_used": ";".join(l for l in C.LABELS if l not in used),
                   "majority_label": C.LABELS[maj], "majority_share": float(shares[maj]),
                   "entropy_bits": h, "entropy_normalised": h / H_MAX,
                   "diag_single_label": len(used) == 1}
            for t in DIAG_THRESHOLDS:
                row[f"diag_majority_ge_{t}"] = bool(shares[maj] >= t)
            rows.append(row)
    cell = pd.DataFrame(rows)
    cell.to_csv(out / "label_usage_by_cell.csv", index=False)

    ck = []
    for run in C.RUNS:
        g = cell[cell["model_key"] == run["model_key"]]
        if g.empty:
            continue
        tot = g[["n_R", "n_U", "n_Ur"]].sum()
        ck.append({
            "model_key": run["model_key"], "model": run["display"], "family": run["family"],
            "n_conditions_usable": len(g), "conditions": ",".join(g["condition"]),
            "n_conditions_partial": int((g["condition_status"] == "partial").sum()),
            "n_single_label": int(g["diag_single_label"].sum()),
            "n_majority_ge_0.99": int(g["diag_majority_ge_0.99"].sum()),
            "n_majority_ge_0.95": int(g["diag_majority_ge_0.95"].sum()),
            "labels_never_used_in_any_condition": ";".join(
                l for l in C.LABELS if tot[f"n_{C.LABEL_SHORT[l]}"] == 0),
            "majority_labels": ";".join(sorted(set(g["majority_label"]))),
            "majority_share_min": g["majority_share"].min(), "majority_share_max": g["majority_share"].max(),
            "entropy_min": g["entropy_bits"].min(), "entropy_max": g["entropy_bits"].max(),
            "descriptive_pattern": checkpoint_pattern(g),
        })
    ck = pd.DataFrame(ck)
    ck.to_csv(out / "label_usage_by_checkpoint.csv", index=False)
    pd.DataFrame([{"diagnostic": "diag_single_label", "definition": "exactly one of R/U/Ur used", "type": "exact"},
                  *[{"diagnostic": f"diag_majority_ge_{t}", "definition": f"majority share >= {t}",
                     "type": "descriptive cut, not a scientific threshold"} for t in DIAG_THRESHOLDS]]
                 ).to_csv(out / "diagnostic_definitions.csv", index=False)

    # ---------------- LaTeX: per cell (entropy and majority share), one row per model
    tab = []
    for run in C.RUNS:
        g = cell[cell["model_key"] == run["model_key"]].set_index("condition")
        if g.empty:
            continue
        r = {"Model": run["display"]}
        for c in C.CONDITIONS:
            if c not in g.index:
                r[c] = "not run"
                continue
            x = g.loc[c]
            txt = f"{x['entropy_bits']:.2f} / {100*x['majority_share']:.1f}{C.LABEL_SHORT[x['majority_label']]}"
            if x["diag_single_label"]:
                txt = r"\textbf{" + txt + r"}$^{s}$"
            elif x["diag_majority_ge_0.99"]:
                txt = r"\textbf{" + txt + "}"
            if x["condition_status"] == "partial":
                txt += r"$^\dagger$"
            r[c] = txt
        r["Never used"] = ck.loc[ck["model_key"] == run["model_key"], "labels_never_used_in_any_condition"].iloc[0] \
            .replace("Reliable", "R").replace("Uncertain", "U").replace("Unreliable", "Ur").replace(";", ", ") or "--"
        tab.append(r)
    C.write_latex_table(
        C.TABLES / "tab_label_usage.tex", pd.DataFrame(tab),
        caption=r"Label usage per model and condition (ground-truth independent): 3-class entropy in bits "
                r"(maximum 1.58) / majority share and majority label. Bold: majority share $\geq 0.99$; "
                r"$^{s}$ a single label is used. These cuts are descriptive diagnostics, not scientific "
                r"thresholds. $^\dagger$ partial condition (valid rows only). ``Never used'': labels absent in "
                r"every available condition.",
        label="tab:label_usage", colspec="llllllc", escape_cells=False)

    ct = ck[["model", "n_conditions_usable", "n_single_label", "n_majority_ge_0.99", "n_majority_ge_0.95",
             "labels_never_used_in_any_condition", "descriptive_pattern"]].copy()
    ct["labels_never_used_in_any_condition"] = ct["labels_never_used_in_any_condition"].replace("", "--")
    ct.columns = ["Model", "Cond.", "Single label", r"Maj.\ $\geq$0.99", r"Maj.\ $\geq$0.95", "Never used",
                  "Descriptive pattern"]
    ct = ct.astype(str)
    ct["Descriptive pattern"] = ct["Descriptive pattern"].str.replace(">=", r"$\geq$", regex=False)
    C.write_latex_table(
        C.TABLES / "tab_label_usage_checkpoint.tex", ct,
        caption=r"Per-checkpoint label-usage summary: number of available conditions, and how many of them use a "
                r"single label or have a majority share $\geq 0.99$ / $\geq 0.95$ (descriptive diagnostics).",
        label="tab:label_usage_checkpoint", colspec="lrrrrll", escape_cells=False)

    # ---------------- Figure: entropy heatmap
    models = [r["display"] for r in C.RUNS if r["display"] in set(cell["model"])]
    mat = cell.pivot(index="model", columns="condition", values="entropy_bits").reindex(index=models,
                                                                                         columns=list(C.CONDITIONS))
    maj = cell.pivot(index="model", columns="condition", values="majority_share").reindex(index=models,
                                                                                          columns=list(C.CONDITIONS))
    fig, ax = plt.subplots(figsize=(5.2, 0.38 * len(models) + 1.2))
    im = ax.imshow(mat.to_numpy(dtype=float), vmin=0, vmax=H_MAX, cmap="magma", aspect="auto")
    for i in range(len(models)):
        for j in range(5):
            v = mat.iloc[i, j]
            if np.isnan(v):
                ax.text(j, i, "not run", ha="center", va="center", fontsize=5.5, color="0.5")
            else:
                ax.text(j, i, f"{v:.2f}\n{100*maj.iloc[i, j]:.0f}%", ha="center", va="center", fontsize=5.5,
                        color="white" if v < 0.9 else "black")
    ax.set_xticks(range(5), C.CONDITIONS)
    ax.set_yticks(range(len(models)), models, fontsize=7)
    fig.colorbar(im, ax=ax, label="entropy (bits)", shrink=0.8)
    ax.set_title("Output entropy and majority share (GT-independent)", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_label_usage_entropy.{ext}", dpi=200)
    plt.close(fig)
    print(ck[["model", "n_conditions_usable", "n_single_label", "n_majority_ge_0.99", "n_majority_ge_0.95",
              "labels_never_used_in_any_condition", "majority_share_min", "majority_share_max",
              "descriptive_pattern"]].round(4).to_string())


if __name__ == "__main__":
    main()
