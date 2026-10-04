#!/usr/bin/env python3
"""Analysis 1 — master results table (one row per model x condition).

Decision-behavior metrics (reference-free) and Protocol-v2 IoU-alignment metrics are
computed from the derived sample-level table and kept in separate column groups and
separate LaTeX tables. Alignment cell names:
  align_R_matched   (legacy TP)   Reliable on a LabelMe-matched detection
  align_R_unmatched (legacy FP)   Reliable on a LabelMe-unmatched detection
  align_Ur_matched  (legacy FN)   Unreliable on a LabelMe-matched detection
  align_Ur_unmatched(legacy TN)   Unreliable on a LabelMe-unmatched detection
Uncertain is excluded from binary metrics. Rows are ordered by the run registry,
not by any metric; no ranking is implied.

Includes full_clean and partial cells (common.CONDITION_POLICY). A partial cell is computed
over its valid rows only; n_samples and condition_status say so. LaTeX: the original
seven-model panel (tab_master_*.tex) and the remaining checkpoints (tab_master_*_extended.tex).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> None:
    C.ensure_dirs()
    df = C.load_samples(include_partial=True)
    audit = pd.read_csv(C.OUT / "audit" / "integrity_checks.csv")
    inv = pd.read_csv(C.OUT / "audit" / "model_inventory.csv").set_index("model_key")

    rows = []
    for run in C.RUNS:
        for cond in C.CONDITIONS:
            sub = df[(df["model_key"] == run["model_key"]) & (df["condition"] == cond)]
            if sub.empty:
                continue
            a = audit[(audit["model_key"] == run["model_key"]) & (audit["condition"] == cond)].iloc[0]
            cc = C.cell_counts(sub)
            m = C.metrics_from_counts({k: np.float64(v) for k, v in cc.items()})
            rows.append({
                "model_key": run["model_key"], "model": run["display"], "family": run["family"],
                "nominal_size": run["nominal_size"],
                "params_billions_counted": inv.loc[run["model_key"], "parameters_billions"],
                "condition": cond,
                "model_conditions_analysed": inv.loc[run["model_key"], "conditions_analysed"],
                "model_conditions_partial": inv.loc[run["model_key"], "conditions_partial"]
                if isinstance(inv.loc[run["model_key"], "conditions_partial"], str) else "",
                "condition_status": a["status"],
                "n_samples": len(sub), "n_invalid_excluded": C.N_COHORT - len(sub),
                "n_parse_error": int(a["n_parse_error"]), "n_inference_error": int(a["n_inference_error"]),
                # --- decision behavior (reference-free)
                "n_reliable": cc["R_m"] + cc["R_u"], "n_uncertain": cc["U_m"] + cc["U_u"],
                "n_unreliable": cc["Ur_m"] + cc["Ur_u"],
                "reliable_rate": m["reliable_rate"], "uncertain_rate": m["uncertain_rate"],
                "unreliable_rate": m["unreliable_rate"],
                "coverage": m["coverage"], "abstention_rate": m["abstention_rate"],
                # --- IoU-alignment (Protocol v2)
                "align_R_matched": cc["R_m"], "align_R_unmatched": cc["R_u"],
                "align_Ur_matched": cc["Ur_m"], "align_Ur_unmatched": cc["Ur_u"],
                "align_U_matched": cc["U_m"], "align_U_unmatched": cc["U_u"],
                "n_decided": cc["R_m"] + cc["R_u"] + cc["Ur_m"] + cc["Ur_u"],
                "n_decided_unmatched": cc["R_u"] + cc["Ur_u"],
                "align_accuracy": m["accuracy"], "align_precision": m["precision"],
                "align_sensitivity": m["sensitivity"], "align_specificity": m["specificity"],
                "align_f1": m["f1"], "align_balanced_accuracy": m["balanced_accuracy"],
            })
    master = pd.DataFrame(rows)

    # Always-Reliable reference row (computed, not typed): every detection answered Reliable.
    ref = df.drop_duplicates("sample_id")
    nm, nu = int(ref["matched_gt"].sum()), int((~ref["matched_gt"]).sum())
    base = C.metrics_from_counts({"R_m": np.float64(nm), "R_u": np.float64(nu), "U_m": 0.0, "U_u": 0.0,
                                  "Ur_m": 0.0, "Ur_u": 0.0})
    baseline = {"model_key": "always_reliable_reference", "model": "Always-Reliable (reference)",
                "condition": "-", "n_samples": C.N_COHORT, "n_reliable": C.N_COHORT, "n_uncertain": 0,
                "n_unreliable": 0, "align_R_matched": nm, "align_R_unmatched": nu, "align_Ur_matched": 0,
                "align_Ur_unmatched": 0, "n_decided": C.N_COHORT, "n_decided_unmatched": nu,
                **{("align_" + k if k in ("accuracy", "precision", "sensitivity", "specificity", "f1",
                                         "balanced_accuracy") else k): v
                   for k, v in base.items() if k != "n"}}
    master_out = pd.concat([master, pd.DataFrame([baseline])], ignore_index=True)
    master_out.to_csv(C.OUT / "master_results.csv", index=False)

    def nlabel(r):
        return C.fmt(int(r["n_samples"])) + (r"$^\dagger$" if r["condition_status"] == "partial" else "")

    core = master[master["model"].isin(C.CORE_PANEL)]
    ext = master[~master["model"].isin(C.CORE_PANEL)]
    behavior_tables(core, C.TABLES / "tab_master_behavior.tex", "tab:master_behavior",
                    r"Decision behavior per model and input condition for the seven-model panel ($N=5{,}747$ "
                    r"each; reference-free). Coverage $=(\mathrm{R}+\mathrm{Ur})/N$. Rows follow the run "
                    r"registry, not a ranking. Further checkpoints are in Table~\ref{tab:master_behavior_extended}.")
    behavior_tables(ext, C.TABLES / "tab_master_behavior_extended.tex", "tab:master_behavior_extended",
                    r"Decision behavior for the further checkpoints (reference-free). $N$ = valid decisions; "
                    r"$^\dagger$ partial condition: parse-error rows are excluded, not repaired. Conditions not "
                    r"yet run are omitted. Coverage $=(\mathrm{R}+\mathrm{Ur})/N$.", with_n=True, nlabel=nlabel)
    alignment_tables(core, C.TABLES / "tab_master_alignment.tex", "tab:master_alignment", nm, nu, base,
                     r"Protocol-v2 IoU-alignment metrics per model and condition (seven-model panel). ")
    alignment_tables(ext, C.TABLES / "tab_master_alignment_extended.tex", "tab:master_alignment_extended", nm, nu,
                     base, r"Protocol-v2 IoU-alignment metrics for the further checkpoints ($^\dagger$ partial "
                     r"condition, valid rows only; -- = undefined, zero denominator). ", nlabel=nlabel)
    print(master_out[["model", "condition", "condition_status", "n_samples", "n_reliable", "n_uncertain",
                      "n_unreliable", "coverage", "align_specificity", "align_balanced_accuracy"]].to_string())


def behavior_tables(b, path, label, caption, with_n=False, nlabel=None):
    tb = pd.DataFrame({
        "Model": b["model"], "Cond.": b["condition"],
        "Reliable": b["n_reliable"].map(C.fmt), "Uncertain": b["n_uncertain"].map(C.fmt),
        "Unreliable": b["n_unreliable"].map(C.fmt),
        "R %": (100 * b["reliable_rate"]).map(lambda x: C.fmt(x, 1)),
        "U %": (100 * b["uncertain_rate"]).map(lambda x: C.fmt(x, 1)),
        "Ur %": (100 * b["unreliable_rate"]).map(lambda x: C.fmt(x, 1)),
        "Coverage": b["coverage"].map(C.fmt),
    })
    if with_n:
        tb.insert(2, "$N$", b.apply(nlabel, axis=1))
    C.write_latex_table(path, tb, caption=caption, label=label,
                        colspec="ll" + "r" * (len(tb.columns) - 2), group_col="Model", escape_cells=False)


def alignment_tables(t, path, label, nm, nu, base, lead, nlabel=None):
    ta = pd.DataFrame({
        "Model": t["model"], "Cond.": t["condition"],
        "Decided": t["n_decided"].map(C.fmt),
        "R/M": t["align_R_matched"].map(C.fmt), "R/UM": t["align_R_unmatched"].map(C.fmt),
        "Ur/M": t["align_Ur_matched"].map(C.fmt), "Ur/UM": t["align_Ur_unmatched"].map(C.fmt),
        "Acc": t["align_accuracy"].map(C.fmt), "Prec": t["align_precision"].map(C.fmt),
        "Sens": t["align_sensitivity"].map(C.fmt), "Spec": t["align_specificity"].map(C.fmt),
        "F1": t["align_f1"].map(C.fmt), "BalAcc": t["align_balanced_accuracy"].map(C.fmt),
    })
    bl = pd.DataFrame([{"Model": "Always-Reliable (ref.)", "Cond.": "--", "Decided": C.fmt(C.N_COHORT),
                        "R/M": C.fmt(nm), "R/UM": C.fmt(nu), "Ur/M": "0", "Ur/UM": "0",
                        "Acc": C.fmt(float(base["accuracy"])), "Prec": C.fmt(float(base["precision"])),
                        "Sens": C.fmt(float(base["sensitivity"])), "Spec": C.fmt(float(base["specificity"])),
                        "F1": C.fmt(float(base["f1"])), "BalAcc": C.fmt(float(base["balanced_accuracy"]))}])
    if nlabel is not None:
        ta.insert(2, "$N$", t.apply(nlabel, axis=1).values)
        bl.insert(2, "$N$", C.fmt(C.N_COHORT))
    ta = pd.concat([ta, bl], ignore_index=True)
    C.write_latex_table(
        path, ta,
        caption=lead + r"M = LabelMe-matched, "
                r"UM = LabelMe-unmatched (IoU $\geq 0.5$, greedy one-to-one). Uncertain excluded. "
                r"R/M, R/UM, Ur/M, Ur/UM correspond to the legacy TP, FP, FN, TN. "
                r"``Spec'' is alignment specificity on LabelMe-unmatched detections, \emph{not} "
                r"semantic non-palm specificity: the human review found no confirmed non-palm among them.",
        label=label, colspec="ll" + "r" * (len(ta.columns) - 2), group_col="Model", escape_cells=False)


if __name__ == "__main__":
    main()
