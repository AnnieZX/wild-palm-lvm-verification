#!/usr/bin/env python3
"""
Purpose:
    Ground-truth-independent A1-A5 VLM behavior analysis. Characterizes how the
    stored Reliable / Uncertain / Unreliable decisions change as the input condition
    changes, on the fixed cohort of 5,747 YOLO detections, using only stored
    prediction JSONs. It does not use LabelMe, IoU matching, GT+/GT-, or any
    correctness metric, and it does not measure whether decisions are semantically
    correct. See docs/A1_A5_BEHAVIOR_ANALYSIS.md.

    Read-only with respect to all experiment outputs: no inference, no re-parsing of
    raw responses (the stored ``decision`` field is used as-is, which is the same
    field Protocol v2 evaluation consumed), and nothing is written outside --out-dir.

Input:
    - outputs/verification_dataset/index.csv                       (cohort sample_ids)
    - outputs/verification/<model>/<experiment>/<A1..A5>/sample_*.json   (decision)
    - outputs/verification/<model>/<experiment>/<A1..A5>/results_index.csv (status)

Output (default outputs/a1_a5_behavior_analysis/):
    - table_b_decision_distribution.csv
    - table_c_transitions_a1_reference.csv
    - pairwise_transitions.csv
    - stability_summary.csv
    - per_detection_decisions.csv
    - tables.md
    - analysis_info.json

Example:
    python scripts/analysis/a1_a5_behavior_analysis.py
    python scripts/visualization/plot_a1_a5_behavior.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import ABLATION_CODES, OUTPUTS_DIR, VERIFICATION_DATASET_INDEX_CSV  # noqa: E402

VERIFICATION_ROOT = OUTPUTS_DIR / "verification"
DEFAULT_OUT_DIR = OUTPUTS_DIR / "a1_a5_behavior_analysis"
EXPECTED_N = 5747
CONDITIONS = tuple(ABLATION_CODES)
REFERENCE = "A1"

# Order along the acceptance -> rejection axis. Used only to name the direction of a
# change; it does not imply that either direction is preferable.
LABELS = ("Reliable", "Uncertain", "Unreliable")
REJECTION_RANK = {label: rank for rank, label in enumerate(LABELS)}

# Every model with complete A1-A5 predictions on the full cohort, verified from the
# stored files (see docs/A1_A5_BEHAVIOR_ANALYSIS.md, "Input audit").
# MiniCPM-V-4.5 and Molmo2-8B have only A1 at full scale and are excluded.
# outputs/verification/qwen2_5_vl/20260708_0020 is the cancelled wrong-path resume
# job 8318954 and must not be used.
MODEL_RUNS: dict[str, tuple[str, str]] = {
    "gemma4": ("Gemma-4-12B-it", "gemma4/20260925_gemma4_A1A5_5747"),
    "glm_4_6v_flash": ("GLM-4.6V-Flash", "glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747"),
    "internvl3_5_hf": ("InternVL3.5-8B-HF", "internvl3_5_hf/20260924_internvl3_5_hf_A1A5_5747"),
    "phi4_multimodal": ("Phi-4-multimodal", "phi4_multimodal/20260921_phi4_A1A5_5747"),
    "qwen2_5_vl": ("Qwen2.5-VL-7B", "qwen/20260708_0020"),
    "qwen3_vl": ("Qwen3-VL-8B", "qwen3_vl/qwen3vl_A1A5_5747"),
    # Model-expansion campaign (docs/MODEL_EXPANSION_CAMPAIGN.md), added as each completes.
    "qwen3_vl_4b": ("Qwen3-VL-4B", "qwen3_vl_4b/20261001_qwen3_vl_4b_A1A5_5747"),
}

PROTECTED_DIRS = (
    VERIFICATION_ROOT,
    OUTPUTS_DIR / "evaluation",
    OUTPUTS_DIR / "evaluation_protocol_v2",
    OUTPUTS_DIR / "verification_dataset",
    OUTPUTS_DIR / "semantic_gt_review",
    OUTPUTS_DIR / "semantic_gt_evaluation",
    OUTPUTS_DIR / "reframing_p0_provenance",
    OUTPUTS_DIR / "diagnostics",
)


class CohortError(RuntimeError):
    """Stored predictions do not form the expected comparable cohort."""


def cell_name(src: str, dst: str) -> str:
    return f"{src.lower()}_to_{dst.lower()}"


TRANSITION_CELLS = tuple(cell_name(a, b) for a in LABELS for b in LABELS)


def assert_safe_output_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    for protected in PROTECTED_DIRS:
        p = protected.resolve()
        if resolved == p or p in resolved.parents or resolved in p.parents:
            raise SystemExit(f"Refusing to write into or above protected directory: {protected}")
    if resolved.name.startswith("verification_ablation_"):
        raise SystemExit(f"Refusing to write into ablation input directory: {out_dir}")


def load_cohort(index_csv: Path, expected_n: int | None = EXPECTED_N) -> list[str]:
    ids = pd.read_csv(index_csv, dtype=str)["sample_id"].tolist()
    if len(ids) != len(set(ids)):
        raise CohortError(f"Duplicate sample_id in {index_csv}")
    if expected_n is not None and len(ids) != expected_n:
        raise CohortError(f"{index_csv}: expected {expected_n} detections, found {len(ids)}")
    return sorted(ids)


def load_condition(condition_dir: Path) -> pd.Series:
    """Return stored decisions (index sample_id) for one condition, with integrity checks."""
    decisions: dict[str, str] = {}
    for path in sorted(condition_dir.glob("sample_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(record.get("sample_id", ""))
        if sample_id != path.stem:
            raise CohortError(f"{path}: sample_id {sample_id!r} does not match filename")
        if sample_id in decisions:
            raise CohortError(f"{condition_dir}: duplicate sample_id {sample_id}")
        for field in ("parse_error", "inference_error"):
            if str(record.get(field, "") or "").strip():
                raise CohortError(f"{path}: non-empty {field}")
        decision = str(record.get("decision", "") or "").strip()
        if decision not in LABELS:
            raise CohortError(f"{path}: decision {decision!r} not in {LABELS}")
        decisions[sample_id] = decision
    if not decisions:
        raise CohortError(f"No prediction JSONs in {condition_dir}")

    index_path = condition_dir / "results_index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
        if index["sample_id"].duplicated().any():
            raise CohortError(f"{index_path}: duplicate sample_id rows")
        if set(index["sample_id"]) != set(decisions):
            raise CohortError(f"{index_path}: sample_ids differ from prediction JSONs")
        if not (index["status"] == "ok").all():
            raise CohortError(f"{index_path}: status other than 'ok'")
    return pd.Series(decisions, name=condition_dir.name).sort_index()


def build_decision_matrix(experiment_dir: Path, cohort: list[str]) -> pd.DataFrame:
    """Decisions for one model as a (sample_id x A1..A5) frame on the exact cohort."""
    columns = {}
    for code in CONDITIONS:
        condition_dir = experiment_dir / code
        if not condition_dir.is_dir():
            raise CohortError(f"Missing condition directory {condition_dir}")
        series = load_condition(condition_dir)
        if list(series.index) != cohort:
            missing = sorted(set(cohort) - set(series.index))[:5]
            extra = sorted(set(series.index) - set(cohort))[:5]
            raise CohortError(
                f"{condition_dir}: cohort mismatch (n={len(series)}, missing e.g. {missing}, extra e.g. {extra})"
            )
        columns[code] = series
    frame = pd.DataFrame(columns)
    frame.index.name = "sample_id"
    return frame


def decisions_fingerprint(series: pd.Series) -> str:
    payload = "".join(f"{sid},{label}\n" for sid, label in series.sort_index().items())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pct(count: int, n: int) -> float:
    return round(100.0 * count / n, 4) if n else 0.0


def decision_distribution(decisions: pd.Series) -> dict:
    n = len(decisions)
    counts = {label: int((decisions == label).sum()) for label in LABELS}
    if sum(counts.values()) != n:
        raise AssertionError("Decision counts do not sum to N")
    row = {"n": n}
    for label in LABELS:
        row[label.lower()] = counts[label]
    for label in LABELS:
        row[f"{label.lower()}_pct"] = pct(counts[label], n)
    row["decided"] = counts["Reliable"] + counts["Unreliable"]
    row["abstention_rate"] = round(counts["Uncertain"] / n, 6)
    return row


def transition_summary(src: pd.Series, dst: pd.Series) -> dict:
    """Paired 3x3 transition counts and directional summaries between two conditions."""
    if not src.index.equals(dst.index):
        raise AssertionError("Paired comparison requires identical sample_id index")
    n = len(src)
    crosstab = pd.crosstab(src, dst).reindex(index=LABELS, columns=LABELS, fill_value=0)
    row: dict = {"n": n}
    for a in LABELS:
        for b in LABELS:
            row[cell_name(a, b)] = int(crosstab.loc[a, b])

    unchanged = sum(row[cell_name(a, a)] for a in LABELS)
    toward_rejection = sum(
        row[cell_name(a, b)] for a in LABELS for b in LABELS if REJECTION_RANK[b] > REJECTION_RANK[a]
    )
    toward_acceptance = sum(
        row[cell_name(a, b)] for a in LABELS for b in LABELS if REJECTION_RANK[b] < REJECTION_RANK[a]
    )
    changed = n - unchanged
    row.update({
        "unchanged": unchanged, "unchanged_pct": pct(unchanged, n),
        "changed": changed, "changed_pct": pct(changed, n),
        "toward_rejection": toward_rejection, "toward_rejection_pct": pct(toward_rejection, n),
        "toward_acceptance": toward_acceptance, "toward_acceptance_pct": pct(toward_acceptance, n),
        "net_toward_rejection": toward_rejection - toward_acceptance,
        "net_toward_rejection_pct": pct(toward_rejection - toward_acceptance, n),
    })
    for label in LABELS:
        entries = sum(row[cell_name(a, label)] for a in LABELS if a != label)
        row[f"changed_into_{label.lower()}"] = entries
    validate_transition_row(row)
    return row


def validate_transition_row(row: dict) -> None:
    n = row["n"]
    if sum(row[c] for c in TRANSITION_CELLS) != n:
        raise AssertionError("Transition matrix does not sum to N")
    if row["unchanged"] + row["changed"] != n:
        raise AssertionError("unchanged + changed != N")
    if row["toward_rejection"] + row["toward_acceptance"] != row["changed"]:
        raise AssertionError("toward_rejection + toward_acceptance != changed")
    explicit_rejection = (row["reliable_to_uncertain"] + row["reliable_to_unreliable"]
                          + row["uncertain_to_unreliable"])
    explicit_acceptance = (row["unreliable_to_uncertain"] + row["unreliable_to_reliable"]
                           + row["uncertain_to_reliable"])
    if row["toward_rejection"] != explicit_rejection or row["toward_acceptance"] != explicit_acceptance:
        raise AssertionError("Directional counts disagree with their cell definitions")
    if sum(row[f"changed_into_{label.lower()}"] for label in LABELS) != row["changed"]:
        raise AssertionError("changed_into_* does not sum to changed")


def per_detection_stability(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix[list(CONDITIONS)]
    n_distinct = values.nunique(axis=1)
    adjacent = sum(
        (values[a] != values[b]).astype(int) for a, b in zip(CONDITIONS, CONDITIONS[1:])
    )
    return pd.DataFrame({"n_distinct_labels": n_distinct, "adjacent_changes": adjacent})


def stability_summary(matrix: pd.DataFrame) -> dict:
    per = per_detection_stability(matrix)
    n = len(per)
    identical = int((per["n_distinct_labels"] == 1).sum())
    two = int((per["n_distinct_labels"] == 2).sum())
    three = int((per["n_distinct_labels"] == 3).sum())
    if identical + two + three != n:
        raise AssertionError("Distinct-label categories do not sum to N")
    row = {
        "n": n,
        "identical_all_5": identical, "identical_all_5_pct": pct(identical, n),
        "exactly_2_labels": two, "exactly_2_labels_pct": pct(two, n),
        "all_3_labels": three, "all_3_labels_pct": pct(three, n),
        "label_variability_rate": round((two + three) / n, 6),
        "adjacent_changes_total": int(per["adjacent_changes"].sum()),
        "adjacent_changes_mean": round(float(per["adjacent_changes"].mean()), 6),
    }
    for k in range(len(CONDITIONS)):
        row[f"detections_with_{k}_adjacent_changes"] = int((per["adjacent_changes"] == k).sum())
    for label in LABELS:
        row[f"always_{label.lower()}"] = int((matrix[list(CONDITIONS)] == label).all(axis=1).sum())
    return row


def compute_all(matrices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    table_b, table_c, pairwise, stability = [], [], [], []
    for key, matrix in matrices.items():
        name = MODEL_RUNS[key][0] if key in MODEL_RUNS else key
        base = {"model_key": key, "model": name}
        for code in CONDITIONS:
            table_b.append({**base, "condition": code, **decision_distribution(matrix[code])})
        for a, b in itertools.combinations(CONDITIONS, 2):
            row = {**base, "comparison": f"{a}->{b}", "from_condition": a, "to_condition": b,
                   **transition_summary(matrix[a], matrix[b])}
            pairwise.append(row)
            if a == REFERENCE:
                table_c.append(row)
        stability.append({**base, **stability_summary(matrix)})
    return {
        "table_b": pd.DataFrame(table_b),
        "table_c": pd.DataFrame(table_c),
        "pairwise": pd.DataFrame(pairwise),
        "stability": pd.DataFrame(stability),
    }


def per_detection_frame(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for key, matrix in matrices.items():
        per = per_detection_stability(matrix)
        part = matrix[list(CONDITIONS)].add_prefix(f"{key}__")
        part[f"{key}__n_distinct_labels"] = per["n_distinct_labels"]
        part[f"{key}__adjacent_changes"] = per["adjacent_changes"]
        parts.append(part)
    return pd.concat(parts, axis=1).reset_index()


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def render_markdown(tables: dict[str, pd.DataFrame]) -> str:
    b = tables["table_b"]
    b_md = pd.DataFrame({
        "Model": b["model"], "Cond.": b["condition"], "N": b["n"],
        "Reliable": b["reliable"], "Uncertain": b["uncertain"], "Unreliable": b["unreliable"],
        "Reliable %": b["reliable_pct"].map("{:.2f}".format),
        "Uncertain %": b["uncertain_pct"].map("{:.2f}".format),
        "Unreliable %": b["unreliable_pct"].map("{:.2f}".format),
        "Decided": b["decided"], "Abstention rate": b["abstention_rate"].map("{:.4f}".format),
    })

    c = tables["table_c"]
    c_md = pd.DataFrame({
        "Model": c["model"], "Comparison": c["comparison"],
        "Unchanged": [f"{u} ({p:.2f}%)" for u, p in zip(c["unchanged"], c["unchanged_pct"])],
        "Changed": [f"{u} ({p:.2f}%)" for u, p in zip(c["changed"], c["changed_pct"])],
        "Toward rejection": [f"{u} ({p:.2f}%)" for u, p in zip(c["toward_rejection"], c["toward_rejection_pct"])],
        "Toward acceptance": [f"{u} ({p:.2f}%)" for u, p in zip(c["toward_acceptance"], c["toward_acceptance_pct"])],
        "Net toward rejection (pp)": c["net_toward_rejection_pct"].map("{:+.2f}".format),
    })
    abbrev = {"Reliable": "R", "Uncertain": "U", "Unreliable": "X"}
    m_md = c[["model", "comparison"]].rename(columns={"model": "Model", "comparison": "Comparison"}).copy()
    for a in LABELS:
        for d in LABELS:
            m_md[f"{abbrev[a]}→{abbrev[d]}"] = c[cell_name(a, d)]

    p = tables["pairwise"]
    p_md = p.pivot(index="model", columns="comparison", values="changed_pct").map("{:.2f}".format)
    p_md = p_md.reindex(list(dict.fromkeys(p["model"])))
    p_md = p_md.reset_index().rename(columns={"model": "Model"})

    s = tables["stability"]
    s_md = pd.DataFrame({
        "Model": s["model"], "N": s["n"],
        "Identical A1–A5": [f"{u} ({p:.2f}%)" for u, p in zip(s["identical_all_5"], s["identical_all_5_pct"])],
        "Exactly 2 labels": [f"{u} ({p:.2f}%)" for u, p in zip(s["exactly_2_labels"], s["exactly_2_labels_pct"])],
        "All 3 labels": [f"{u} ({p:.2f}%)" for u, p in zip(s["all_3_labels"], s["all_3_labels_pct"])],
        "Adjacent changes (total)": s["adjacent_changes_total"],
        "Mean adjacent changes / detection": s["adjacent_changes_mean"].map("{:.3f}".format),
        "Always R": s["always_reliable"], "Always U": s["always_uncertain"], "Always X": s["always_unreliable"],
    })

    return "\n\n".join([
        "# A1–A5 VLM behavior analysis (ground-truth independent)",
        "Generated by `scripts/analysis/a1_a5_behavior_analysis.py`. This analysis is ground-truth "
        "independent. It characterizes how VLM decisions change as the visual input condition changes; "
        "it does not measure whether those decisions are semantically correct.",
        "Direction terms follow the ordering Reliable → Uncertain → Unreliable: *toward rejection* = "
        "R→U, R→X, U→X; *toward acceptance* = X→U, X→R, U→R. Neither direction is preferable. "
        "Abbreviations: R = Reliable, U = Uncertain, X = Unreliable.",
        "## Table B — Decision distributions", _md_table(b_md),
        "## Table C — Paired transitions, A1 reference", _md_table(c_md),
        "### Table C (cont.) — 3×3 transition counts (row label = A1 decision)", _md_table(m_md),
        "## Pairwise extension — changed % for every condition pair", _md_table(p_md),
        "## Per-detection stability across A1–A5", _md_table(s_md),
        "Adjacent changes are counted along A1→A2→A3→A4→A5. The conditions are not a single ordered "
        "dimension, so this count depends on the chosen order; the distinct-label counts do not.",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_safe_output_dir(args.out_dir)
    cohort = load_cohort(VERIFICATION_DATASET_INDEX_CSV)

    matrices: dict[str, pd.DataFrame] = {}
    inputs: dict[str, dict] = {}
    for key, (name, rel) in MODEL_RUNS.items():
        experiment_dir = VERIFICATION_ROOT / rel
        print(f"Loading {name}: {experiment_dir.relative_to(PROJECT_ROOT)}")
        matrix = build_decision_matrix(experiment_dir, cohort)
        matrices[key] = matrix
        inputs[key] = {
            "model": name,
            "experiment_dir": str(experiment_dir.relative_to(PROJECT_ROOT)),
            "decision_sha256": {code: decisions_fingerprint(matrix[code]) for code in CONDITIONS},
        }

    tables = compute_all(matrices)
    b = tables["table_b"]
    if not ((b[["reliable", "uncertain", "unreliable"]].sum(axis=1) == EXPECTED_N) & (b["n"] == EXPECTED_N)).all():
        raise AssertionError("Table B rows do not sum to the cohort size")
    for name in ("table_c", "pairwise"):
        if not (tables[name][list(TRANSITION_CELLS)].sum(axis=1) == EXPECTED_N).all():
            raise AssertionError(f"{name}: transition matrices do not sum to the cohort size")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tables["table_b"].to_csv(out / "table_b_decision_distribution.csv", index=False)
    tables["table_c"].to_csv(out / "table_c_transitions_a1_reference.csv", index=False)
    tables["pairwise"].to_csv(out / "pairwise_transitions.csv", index=False)
    tables["stability"].to_csv(out / "stability_summary.csv", index=False)
    per_detection_frame(matrices).to_csv(out / "per_detection_decisions.csv", index=False)
    (out / "tables.md").write_text(render_markdown(tables), encoding="utf-8")
    info = {
        "description": "Ground-truth-independent A1-A5 behavior analysis; see docs/A1_A5_BEHAVIOR_ANALYSIS.md",
        "cohort_index": str(VERIFICATION_DATASET_INDEX_CSV.relative_to(PROJECT_ROOT)),
        "cohort_n": len(cohort),
        "conditions": list(CONDITIONS),
        "reference_condition": REFERENCE,
        "labels": list(LABELS),
        "direction_definition": {
            "toward_rejection": ["Reliable->Uncertain", "Reliable->Unreliable", "Uncertain->Unreliable"],
            "toward_acceptance": ["Unreliable->Uncertain", "Unreliable->Reliable", "Uncertain->Reliable"],
        },
        "excluded_runs": {
            "minicpm_v4_5/20260923_minicpm_A1A5_5747": "A1 only at full scale",
            "molmo2_8b/20260923_molmo2_A1A5_5747": "A1 only at full scale",
            "qwen2_5_vl/20260708_0020": "cancelled wrong-path resume job 8318954 (partial A5, not for use)",
        },
        "inputs": inputs,
    }
    (out / "analysis_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
