#!/usr/bin/env python3
"""Audit every candidate N=5,747 run and build the derived sample-level table.

Integrity checks per model x condition:
  1. exactly 5,747 sample_*.json files;
  2. sample_id unique and equal to the file stem;
  3. sample_id set equals outputs/verification_dataset/index.csv (no missing / extra IDs);
  4. parse_error / inference_error / invalid-decision counts (must be 0);
  5. results_index.csv present, same IDs, all status == ok;
  6. JSON metadata (model_key / condition / experiment_id, when present) match the directory;
  7. Protocol-v2 evaluation CSV present, same IDs, verification_label == JSON decision for
     every sample, image_name equals the cohort index;
  8. A*_metrics.json counts and metrics equal values recomputed from the sample level.
Cross-run check: matched_gt and max_iou are identical across every usable evaluation CSV.

Condition status (policy documented in common.CONDITION_POLICY):
  full_clean  every check passes (the only cells with analysed = True);
  partial     all 5,747 files present and the only defects are parse errors that are
              consistently recorded (empty decision, results_index "parse_error" on exactly
              those IDs, empty Protocol-v2 label, stored metrics equal to a recomputation over
              the valid rows; an undefined metric stored as 0 by the evaluator is noted, not
              counted as a mismatch);
  failed / incomplete / not_run otherwise.
The derived sample table holds valid-decision rows of usable (full_clean or partial) cells
only, with columns condition_status, analysed (= full_clean) and usable. Parse-error rows are
never repaired or imputed. As a labelled diagnostic only, the label string that appears first
after "decision" in each parse-error raw response is counted; it is not used in any analysis.

Reads only; writes paper/analysis/{outputs/audit,derived}/.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


RAW_DECISION = re.compile(r'"decision"\s*:\s*"(Reliable|Uncertain|Unreliable)"')


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        d = json.load(f)
    gen = d.get("generation") or {}
    raw_label = ""
    if d.get("parse_error"):
        m = RAW_DECISION.search(d.get("raw_response") or "")
        raw_label = m.group(1) if m else "none_found"
    return {
        "raw_label_diagnostic": raw_label,
        "file_stem": path.stem,
        "sample_id": d.get("sample_id"),
        "decision": d.get("decision") or "",
        "parse_error": d.get("parse_error") or "",
        "inference_error": d.get("inference_error") or "",
        "json_model_key": d.get("model_key"),
        "json_condition": d.get("condition"),
        "json_experiment_id": d.get("experiment_id"),
        "model_name": d.get("model_name"),
        "checkpoint_revision": gen.get("checkpoint_revision"),
    }


def audit_condition(run: dict, cond: str, index: pd.DataFrame, pool: ThreadPoolExecutor):
    pred_dir = C.VERIFICATION_ROOT / run["pred_rel"] / cond
    eval_dir = C.EVAL_V2_ROOT / run["eval_rel"] / cond
    row = {"model_key": run["model_key"], "display": run["display"], "condition": cond,
           "pred_dir": str(pred_dir.relative_to(C.REPO)), "eval_dir": str(eval_dir.relative_to(C.REPO))}
    issues: list[str] = []      # defects that make a cell unusable
    soft: list[str] = []        # consistently recorded parse errors (cell may be "partial")
    if not pred_dir.is_dir():
        row.update(n_json=0, status="not_run", analysed=False, usable=False, issues="prediction directory absent")
        return row, None
    files = sorted(pred_dir.glob("sample_*.json"))
    row["n_json"] = len(files)
    if len(files) != C.N_COHORT:
        row.update(status="incomplete", analysed=False, usable=False,
                   issues=f"{len(files)} of {C.N_COHORT} prediction files (run in progress or partial)")
        return row, None

    recs = pd.DataFrame(list(pool.map(read_json, files)))
    expected_ids = set(index["sample_id"])
    row["ids_unique"] = bool(recs["sample_id"].is_unique)
    row["ids_match_filenames"] = bool((recs["sample_id"] == recs["file_stem"]).all())
    row["ids_equal_cohort"] = set(recs["sample_id"]) == expected_ids
    row["n_missing_ids"] = len(expected_ids - set(recs["sample_id"]))
    row["n_extra_ids"] = len(set(recs["sample_id"]) - expected_ids)
    row["n_parse_error"] = int((recs["parse_error"] != "").sum())
    row["n_inference_error"] = int((recs["inference_error"] != "").sum())
    row["n_invalid_decision"] = int((~recs["decision"].isin(C.LABELS)).sum())
    parse_ids = set(recs.loc[recs["parse_error"] != "", "sample_id"])
    invalid_ids = set(recs.loc[~recs["decision"].isin(C.LABELS), "sample_id"])
    row["n_valid"] = C.N_COHORT - len(invalid_ids)
    row["parse_error_ids"] = ";".join(sorted(parse_ids)) if len(parse_ids) <= 5 else f"{len(parse_ids)} ids"
    diag = recs.loc[recs["parse_error"] != "", "raw_label_diagnostic"].value_counts()
    row["diagnostic_raw_label_in_parse_errors"] = ";".join(f"{k}:{v}" for k, v in diag.items())
    parse_only = (row["n_inference_error"] == 0 and invalid_ids == parse_ids
                  and (recs.loc[recs["sample_id"].isin(parse_ids), "decision"] == "").all())
    for k, ok in (("ids_unique", row["ids_unique"]), ("ids_match_filenames", row["ids_match_filenames"]),
                  ("ids_equal_cohort", row["ids_equal_cohort"])):
        if not ok:
            issues.append(k + " failed")
    for k in ("n_parse_error", "n_inference_error", "n_invalid_decision"):
        if row[k]:
            (soft if parse_only and k != "n_inference_error" else issues).append(f"{k}={row[k]}")

    # results_index.csv
    ri_path = pred_dir / "results_index.csv"
    if ri_path.is_file():
        ri = pd.read_csv(ri_path, dtype=str)
        row["results_index_rows"] = len(ri)
        row["results_index_all_ok"] = bool((ri["status"] == "ok").all())
        row["results_index_ids_equal"] = set(ri["sample_id"]) == set(recs["sample_id"]) and ri["sample_id"].is_unique
        not_ok = ri[ri["status"] != "ok"]
        ri_consistent = set(not_ok["sample_id"]) == parse_ids and (not_ok["status"] == "parse_error").all()
        if not row["results_index_ids_equal"]:
            issues.append("results_index IDs differ")
        elif not row["results_index_all_ok"]:
            if parse_only and ri_consistent:
                soft.append("results_index marks exactly the parse-error IDs")
            else:
                issues.append("results_index non-ok status not explained by parse errors")
    else:
        row["results_index_rows"] = 0
        issues.append("results_index.csv missing")

    # JSON metadata vs directory (fields absent in legacy records are not an error)
    def meta_mismatch(col, expected):
        present = recs[col].notna()
        return int((present & (recs[col] != expected)).sum()), int(present.sum())
    mk_bad, mk_present = meta_mismatch("json_model_key", run["model_key"])
    cd_bad, cd_present = meta_mismatch("json_condition", cond)
    ex_bad, ex_present = meta_mismatch("json_experiment_id", Path(run["pred_rel"]).name)
    row.update(json_model_key_present=mk_present, json_model_key_mismatch=mk_bad,
               json_condition_present=cd_present, json_condition_mismatch=cd_bad,
               json_experiment_present=ex_present, json_experiment_mismatch=ex_bad)
    if mk_bad or cd_bad or ex_bad:
        issues.append("JSON metadata disagrees with directory")
    row["model_name"] = recs["model_name"].dropna().iloc[0] if recs["model_name"].notna().any() else None
    revs = recs["checkpoint_revision"].dropna().unique()
    row["checkpoint_revision"] = ";".join(revs) if len(revs) else None

    # Protocol-v2 evaluation CSV
    ev_path = eval_dir / f"{cond}_evaluation.csv"
    if not ev_path.is_file():
        issues.append("Protocol-v2 evaluation CSV missing")
        row.update(status="failed", analysed=False, usable=False, issues="; ".join(issues))
        return row, None
    ev = pd.read_csv(ev_path, dtype={"sample_id": str, "image_name": str})
    row["eval_rows"] = len(ev)
    merged = recs.merge(ev, on="sample_id", how="outer", indicator=True, validate="one_to_one")
    row["eval_ids_equal"] = bool((merged["_merge"] == "both").all())
    both = merged[merged["_merge"] == "both"]
    row["eval_label_mismatches"] = int((both["verification_label"].fillna("") != both["decision"]).sum())
    row["eval_ablation_field_ok"] = bool((ev["ablation"] == cond).all())
    idx_img = index.set_index("sample_id")["image_name"]
    row["eval_image_name_mismatches"] = int((both["image_name"].values != idx_img.loc[both["sample_id"]].values).sum())
    if not row["eval_ids_equal"]:
        issues.append("evaluation IDs differ from predictions")
    if row["eval_label_mismatches"]:
        issues.append(f"{row['eval_label_mismatches']} evaluation labels differ from prediction decisions")
    if not row["eval_ablation_field_ok"]:
        issues.append("evaluation ablation field differs from condition")
    if row["eval_image_name_mismatches"]:
        issues.append("evaluation image_name differs from cohort index")

    sample = both[["sample_id", "image_name", "decision", "matched_gt", "max_iou", "yolo_confidence"]].copy()
    sample["matched_gt"] = sample["matched_gt"].astype(str).str.lower().eq("true")

    # metrics JSON vs recomputation
    m_path = eval_dir / f"{cond}_metrics.json"
    if m_path.is_file():
        stored = json.loads(m_path.read_text())
        cc = C.cell_counts(sample)
        mets = C.metrics_from_counts({k: np.float64(v) for k, v in cc.items()})
        checks = {
            "true_positive": cc["R_m"], "false_positive": cc["R_u"], "false_negative": cc["Ur_m"],
            "true_negative": cc["Ur_u"], "reliable_count": cc["R_m"] + cc["R_u"],
            "uncertain_count": cc["U_m"] + cc["U_u"], "unreliable_count": cc["Ur_m"] + cc["Ur_u"],
        }
        bad = [k for k, v in checks.items() if stored.get(k) != v]
        undefined_as_zero = []
        for k_json, k_ours in (("accuracy", "accuracy"), ("precision", "precision"), ("recall", "sensitivity"),
                               ("specificity", "specificity"), ("f1", "f1"),
                               ("balanced_accuracy", "balanced_accuracy")):
            sv = stored.get(k_json)
            ov = float(mets[k_ours])
            if np.isnan(ov) and sv == 0:
                undefined_as_zero.append(k_json)
            elif sv is None or abs(sv - ov) > 6e-5:
                bad.append(k_json)
        if stored.get("evaluated_samples") is not None and stored["evaluated_samples"] != len(sample[sample["decision"].isin(C.LABELS)]):
            bad.append("evaluated_samples")
        row["metrics_json_mismatch"] = ";".join(bad)
        row["metrics_json_undefined_stored_as_zero"] = ";".join(undefined_as_zero)
        row["metrics_json_protocol"] = stored.get("evaluation_protocol")
        if bad:
            issues.append("metrics JSON differs from recomputation: " + ",".join(bad))
        if stored.get("evaluation_protocol") != "v2":
            issues.append("metrics JSON not Protocol v2")
    else:
        issues.append("metrics JSON missing")

    row["issues"] = "; ".join(issues + soft)
    row["status"] = "failed" if issues else ("partial" if soft else "full_clean")
    row["analysed"] = row["status"] == "full_clean"
    row["usable"] = row["status"] in ("full_clean", "partial")
    if not row["usable"]:
        return row, None
    sample = sample[sample["decision"].isin(C.LABELS)].copy()
    sample.insert(0, "condition", cond)
    sample.insert(0, "display", run["display"])
    sample.insert(0, "model_key", run["model_key"])
    return row, sample


def main() -> None:
    C.ensure_dirs()
    audit_dir = C.OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(C.INDEX_CSV, dtype={"sample_id": str, "image_name": str})
    assert len(index) == C.N_COHORT and index["sample_id"].is_unique

    rows, samples = [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for run in C.RUNS:
            for cond in C.CONDITIONS:
                row, sample = audit_condition(run, cond, index, pool)
                row["expected_status"] = run["expected"]
                rows.append(row)
                if sample is not None:
                    sample["condition_status"] = row["status"]
                    sample["analysed"] = row["analysed"]
                    sample["usable"] = row["usable"]
                    samples.append(sample)
                print(f"{run['display']:<18} {cond}  {row['status']:<10} {row.get('issues', '')}")

    audit = pd.DataFrame(rows)
    df = pd.concat(samples, ignore_index=True)
    df = df.rename(columns={"image_name": "image_id"})

    # Cross-run check: alignment labels and IoU identical in every usable evaluation CSV
    ok = df
    per_sample = ok.groupby("sample_id").agg(matched_n=("matched_gt", "nunique"), iou_n=("max_iou", "nunique"))
    cross_matched_ok = bool((per_sample["matched_n"] == 1).all())
    cross_iou_ok = bool((per_sample["iou_n"] == 1).all())
    ref = ok.drop_duplicates("sample_id")
    n_matched = int(ref["matched_gt"].sum())
    n_unmatched = int((~ref["matched_gt"]).sum())

    df.to_csv(C.SAMPLE_TABLE, index=False, compression="gzip")
    audit.to_csv(audit_dir / "integrity_checks.csv", index=False)

    # model inventory with parameter counts from checkpoint headers
    inv = []
    for run in C.RUNS:
        ck = C.MODELS_ROOT / run["checkpoint"]
        n_params = C.count_safetensors_parameters(ck) if ck.is_dir() else None
        sub = audit[audit["model_key"] == run["model_key"]]
        meta_txt = ck / "DOWNLOAD_META.txt"
        inv.append({
            "model_key": run["model_key"], "display": run["display"], "family": run["family"],
            "nominal_size_from_name": run["nominal_size"],
            "parameters_counted_from_checkpoint": n_params,
            "parameters_billions": round(n_params / 1e9, 2) if n_params else None,
            "checkpoint_dir": str(ck),
            "checkpoint_revision_in_records": ";".join(sorted(set(sub["checkpoint_revision"].dropna()))) or None,
            "download_meta_present": meta_txt.is_file(),
            "conditions_analysed": ",".join(sub.loc[sub["analysed"].fillna(False).astype(bool), "condition"]),
            "conditions_partial": ",".join(sub.loc[sub["status"] == "partial", "condition"]),
            "condition_policy": C.CONDITION_POLICY.get(run["model_key"], ""),
            "conditions_files_present": ",".join(f"{r.condition}:{r.n_json}" for r in sub.itertuples() if r.n_json),
            "expected_status": run["expected"],
        })
    inv = pd.DataFrame(inv)
    inv.to_csv(audit_dir / "model_inventory.csv", index=False)

    info = {
        "cohort_n": C.N_COHORT,
        "n_images": int(index["image_name"].nunique()),
        "analysed_cells": int(audit["analysed"].fillna(False).astype(bool).sum()),
        "full_clean_cells": int((audit["status"] == "full_clean").sum()),
        "partial_cells": int((audit["status"] == "partial").sum()),
        "status_counts": audit["status"].value_counts().to_dict(),
        "condition_policy": C.CONDITION_POLICY,
        "cross_run_matched_gt_identical": cross_matched_ok,
        "cross_run_max_iou_identical": cross_iou_ok,
        "labelme_matched": n_matched, "labelme_unmatched": n_unmatched,
        "known_exclusions": C.KNOWN_EXCLUSIONS,
    }
    (audit_dir / "audit_info.json").write_text(json.dumps(info, indent=2))

    lines = ["# Integrity audit (auto-generated by `a00_audit.py`)", "",
             f"- Cohort: {C.N_COHORT} detections on {info['n_images']} images.",
             f"- full_clean cells (analysed): **{info['full_clean_cells']}**; partial cells (valid rows only): "
             f"**{info['partial_cells']}**. Status counts: {info['status_counts']}.",
             f"- `matched_gt` identical across all analysed evaluation CSVs: **{cross_matched_ok}**; "
             f"`max_iou` identical: **{cross_iou_ok}** (over all usable cells).",
             f"- LabelMe-matched / unmatched: {n_matched} / {n_unmatched}.", "",
             "| Model | Cond | Files | Valid | Parse err | Infer err | Invalid | Eval label mismatches | Status | Issues |",
             "|---|---|--:|--:|--:|--:|--:|--:|---|---|"]
    for r in audit.itertuples():
        g = lambda k: getattr(r, k, "") if not (isinstance(getattr(r, k, ""), float) and np.isnan(getattr(r, k, ""))) else ""
        lines.append(f"| {r.display} | {r.condition} | {r.n_json} | {g('n_valid')} | {g('n_parse_error')} | {g('n_inference_error')} | "
                     f"{g('n_invalid_decision')} | {g('eval_label_mismatches')} | {r.status} | {g('issues')} |")
    lines += ["", "Condition policy (common.CONDITION_POLICY):", ""]
    lines += [f"- `{k}`: {v}" for k, v in C.CONDITION_POLICY.items()]
    lines += ["", "Diagnostic only (not used in any analysis): label string found first after "
              "`\"decision\"` in parse-error raw responses:", ""]
    for r in audit[audit.get("diagnostic_raw_label_in_parse_errors", pd.Series(dtype=str)).fillna("") != ""].itertuples():
        lines.append(f"- {r.display} {r.condition}: {r.diagnostic_raw_label_in_parse_errors}")
    lines += ["", "Known exclusions (on disk, deliberately not analysed):", ""]
    lines += [f"- `{k}`: {v}" for k, v in C.KNOWN_EXCLUSIONS.items()]
    (audit_dir / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
