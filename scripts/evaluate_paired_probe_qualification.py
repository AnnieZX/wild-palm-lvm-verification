#!/usr/bin/env python3
"""
Technical qualification verdict for one model on the fixed GT-independent paired probe.

Reads only stored probe predictions (no inference, no GT, no re-parsing):
    <experiment-dir>/<A1..A5>/sample_*.json, results_index.csv
    <experiment-dir>/determinism_A1/sample_*.json

Hard gates (technical only):
    - every condition produced exactly the probe sample IDs (model loaded, ran)
    - zero inference errors (includes OOM)
    - usable parsed output >= 95% in every condition
    - every parsed decision is Reliable | Uncertain | Unreliable
    - zero garbage outputs (empty response or no JSON object start)
    - checkpoint revision recorded in records equals the config pin (when pinned)
    - one resolved attention backend across all conditions

Reported, never gated (behavioral findings): decision distribution, label
dominance, Uncertain usage, paired A1->Ak transitions, cross-condition
responsiveness, determinism re-run agreement, runtime, peak GPU memory.

Writes qualification_report.json, qualification_report.md and
QUALIFICATION_VERDICT.txt (PASS | FAIL) to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

CONDITIONS = ("A1", "A2", "A3", "A4", "A5")
CONDITION_FOLDERS = {
    "A1": "A1_overlay_only",
    "A2": "A2_overlay_confidence",
    "A3": "A3_overlay_confidence_geometry",
    "A4": "A4_overlay_crop_confidence",
    "A5": "A5_crop_only",
}
LABELS = ("Reliable", "Uncertain", "Unreliable")
USABLE_MIN = 0.95
FULL_COHORT_N = 5747


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired-probe technical qualification verdict.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--gpu-mem-log", type=Path, default=None)
    parser.add_argument(
        "--reference-experiment-dir",
        type=Path,
        action="append",
        default=[],
        help="Stored run(s) of the same model; probe agreement on shared IDs is reported (not gated)",
    )
    return parser.parse_args()


def reference_agreement(experiment_dir: Path, reference_dir: Path) -> dict:
    """Per-condition agreement between probe records and a stored run on shared sample IDs."""
    result: dict[str, dict] = {}
    for code in CONDITIONS:
        ref_dir = reference_dir / code
        if not ref_dir.is_dir():
            continue
        probe = load_condition(experiment_dir / code)
        shared = [s for s in probe if (ref_dir / f"{s}.json").is_file()]
        if not shared:
            continue
        ref = {s: json.loads((ref_dir / f"{s}.json").read_text(encoding="utf-8")) for s in shared}
        result[code] = {
            "n_shared": len(shared),
            "decision_identical": sum(probe[s].get("decision") == ref[s].get("decision") for s in shared),
            "raw_identical": sum(probe[s].get("raw_response") == ref[s].get("raw_response") for s in shared),
        }
    return result


def classify_record(record: dict) -> str:
    """One of: usable, inference_error, invalid_label, garbage, truncated, parse_error."""
    if str(record.get("inference_error") or "").strip():
        return "inference_error"
    decision = str(record.get("decision") or "")
    parse_error = str(record.get("parse_error") or "").strip()
    if not parse_error:
        return "usable" if decision in LABELS else "invalid_label"
    raw = str(record.get("raw_response") or "").strip()
    if not raw or "{" not in raw:
        return "garbage"
    if "}" not in raw[raw.index("{"):]:
        return "truncated"
    return "parse_error"


def load_condition(condition_dir: Path) -> dict[str, dict]:
    records = {}
    for path in sorted(condition_dir.glob("sample_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record["sample_id"])] = record
    return records


def probe_ids(probe_root: Path, code: str) -> list[str]:
    index = pd.read_csv(probe_root / CONDITION_FOLDERS[code] / "prompt_index.csv")
    return index["sample_id"].astype(str).tolist()


def peak_gpu_mib(log_path: Path | None) -> int | None:
    if log_path is None or not log_path.is_file():
        return None
    peak = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                value = int(parts[2].replace("MiB", "").strip())
            except ValueError:
                continue
            peak = value if peak is None else max(peak, value)
    return peak


def evaluate(experiment_dir: Path, probe_root: Path, config: dict, gpu_mem_log: Path | None) -> dict:
    expected_ids = probe_ids(probe_root, "A1")
    failures: list[str] = []
    warnings: list[str] = []
    per_condition: dict[str, dict] = {}
    decisions: dict[str, dict[str, str]] = {}
    backends: set[str] = set()
    revisions: set[str] = set()

    for code in CONDITIONS:
        ids = probe_ids(probe_root, code)
        if ids != expected_ids:
            failures.append(f"{code}: probe sample IDs differ from A1")
        records = load_condition(experiment_dir / code)
        missing = sorted(set(ids) - set(records))
        categories = Counter(classify_record(records[s]) for s in ids if s in records)
        usable = categories.get("usable", 0)
        runtimes = [float(records[s].get("runtime_seconds") or 0) for s in ids if s in records]
        decision_counts = Counter(
            records[s]["decision"] for s in ids if s in records and classify_record(records[s]) == "usable"
        )
        decisions[code] = {
            s: records[s]["decision"] for s in ids if s in records and classify_record(records[s]) == "usable"
        }
        for s in ids:
            generation = (records.get(s) or {}).get("generation") or {}
            if generation.get("attn_implementation_resolved"):
                backends.add(str(generation["attn_implementation_resolved"]))
            if generation.get("checkpoint_revision"):
                revisions.add(str(generation["checkpoint_revision"]))

        usable_rate = usable / len(ids) if ids else 0.0
        mean_rt = statistics.mean(runtimes) if runtimes else None
        per_condition[code] = {
            "n_expected": len(ids),
            "n_records": len(records),
            "missing": len(missing),
            "categories": dict(categories),
            "usable_rate": round(usable_rate, 4),
            "decision_counts": {label: decision_counts.get(label, 0) for label in LABELS},
            "runtime_mean_s": round(mean_rt, 3) if mean_rt is not None else None,
            "runtime_median_s": round(statistics.median(runtimes), 3) if runtimes else None,
            "projected_full_hours": round(mean_rt * FULL_COHORT_N / 3600, 2) if mean_rt else None,
        }

        if missing:
            failures.append(f"{code}: {len(missing)} probe samples have no record")
        if categories.get("inference_error"):
            failures.append(f"{code}: {categories['inference_error']} inference errors")
        if categories.get("invalid_label"):
            failures.append(f"{code}: {categories['invalid_label']} parsed records with invalid labels")
        if categories.get("garbage"):
            failures.append(f"{code}: {categories['garbage']} garbage outputs")
        if usable_rate < USABLE_MIN:
            failures.append(f"{code}: usable parsed output {usable_rate:.1%} < {USABLE_MIN:.0%}")
        if categories.get("truncated"):
            warnings.append(f"{code}: {categories['truncated']} truncated outputs (counted unusable)")

    pinned = str(config.get("revision") or "")
    if pinned:
        if revisions != {pinned}:
            failures.append(f"checkpoint revision in records {sorted(revisions)} != config pin {pinned}")
    if len(backends) > 1:
        failures.append(f"multiple resolved attention backends: {sorted(backends)}")
    configured_backend = config.get("attn_implementation")
    if configured_backend and backends and backends != {str(configured_backend)}:
        failures.append(f"resolved backend {sorted(backends)} != configured {configured_backend}")

    # Behavioral descriptors (never gated).
    paired_ids = [s for s in expected_ids if all(s in decisions[c] for c in CONDITIONS)]
    transitions = {}
    for code in CONDITIONS[1:]:
        changed = sum(decisions["A1"][s] != decisions[code][s] for s in paired_ids)
        matrix = Counter(f"{decisions['A1'][s]}->{decisions[code][s]}" for s in paired_ids)
        transitions[f"A1->{code}"] = {
            "n_paired": len(paired_ids),
            "changed": changed,
            "change_rate": round(changed / len(paired_ids), 4) if paired_ids else None,
            "matrix": dict(sorted(matrix.items())),
        }
    responsive = sum(len({decisions[c][s] for c in CONDITIONS}) > 1 for s in paired_ids)
    total_counts = Counter()
    for code in CONDITIONS:
        total_counts.update(per_condition[code]["decision_counts"])
    total_usable = sum(total_counts.values())
    dominant_label, dominant_n = (total_counts.most_common(1)[0] if total_usable else ("", 0))
    behavior_notes = []
    if total_usable and dominant_n / total_usable >= 0.95:
        behavior_notes.append(f"{dominant_label} dominates ({dominant_n}/{total_usable})")
    if total_usable and total_counts.get("Uncertain", 0) == 0:
        behavior_notes.append("Uncertain unused")
    if paired_ids and responsive == 0:
        behavior_notes.append("no decision changes across A1-A5")

    determinism = {}
    det_records = load_condition(experiment_dir / "determinism_A1")
    if det_records:
        a1_records = load_condition(experiment_dir / "A1")
        compared = [s for s in det_records if s in a1_records]
        identical = sum(det_records[s].get("raw_response") == a1_records[s].get("raw_response") for s in compared)
        same_decision = sum(det_records[s].get("decision") == a1_records[s].get("decision") for s in compared)
        determinism = {
            "n_compared": len(compared),
            "raw_identical": identical,
            "decision_identical": same_decision,
        }
        if identical != len(compared):
            warnings.append(f"determinism re-run: {identical}/{len(compared)} raw responses identical")
    else:
        warnings.append("no determinism re-run records found")

    all_usable = sum(per_condition[c]["categories"].get("usable", 0) for c in CONDITIONS)
    all_expected = sum(per_condition[c]["n_expected"] for c in CONDITIONS)
    return {
        "verdict": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "model_key": config.get("registry_key", ""),
        "model_label": config.get("model_label", ""),
        "hf_repo": config.get("hf_repo", ""),
        "config_revision": pinned,
        "record_revisions": sorted(revisions),
        "attn_implementation_config": configured_backend,
        "attn_implementation_resolved": sorted(backends),
        "experiment_dir": str(experiment_dir),
        "probe_root": str(probe_root),
        "probe_sample_ids": f"{expected_ids[0]}..{expected_ids[-1]} (n={len(expected_ids)})",
        "overall_usable_rate": round(all_usable / all_expected, 4) if all_expected else 0.0,
        "per_condition": per_condition,
        "behavior": {
            "decision_totals": dict(total_counts),
            "paired_transitions": transitions,
            "responsive_samples": responsive,
            "n_paired": len(paired_ids),
            "notes": behavior_notes,
        },
        "determinism": determinism,
        "peak_gpu_mib": peak_gpu_mib(gpu_mem_log),
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Paired-probe technical qualification — {report['model_label'] or report['model_key']}",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        f"- HF repo / revision: `{report['hf_repo']}` @ `{report['config_revision']}`",
        f"- Attention backend: config `{report['attn_implementation_config']}`, "
        f"resolved `{', '.join(report['attn_implementation_resolved']) or 'unrecorded'}`",
        f"- Probe: {report['probe_sample_ids']} × A1–A5",
        f"- Overall usable parsed output: {report['overall_usable_rate']:.1%}",
        f"- Peak GPU memory: {report['peak_gpu_mib']} MiB",
        "",
        "| Cond | usable | R | U | Ur | non-usable | mean s | proj. full h |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for code, row in report["per_condition"].items():
        non_usable = {k: v for k, v in row["categories"].items() if k != "usable"}
        d = row["decision_counts"]
        lines.append(
            f"| {code} | {row['usable_rate']:.0%} | {d['Reliable']} | {d['Uncertain']} | {d['Unreliable']} | "
            f"{non_usable or '—'} | {row['runtime_mean_s']} | {row['projected_full_hours']} |"
        )
    behavior = report["behavior"]
    lines += ["", "## Behavior (reported, not gated)", ""]
    lines.append(f"- Samples whose decision changes across A1–A5: {behavior['responsive_samples']}/{behavior['n_paired']}")
    for key, value in behavior["paired_transitions"].items():
        lines.append(f"- {key}: {value['changed']}/{value['n_paired']} changed")
    for note in behavior["notes"]:
        lines.append(f"- {note}")
    if report["determinism"]:
        det = report["determinism"]
        lines.append(
            f"- Determinism re-run (A1): raw identical {det['raw_identical']}/{det['n_compared']}, "
            f"decision identical {det['decision_identical']}/{det['n_compared']}"
        )
    for ref, per_condition in (report.get("reference_agreement") or {}).items():
        lines += ["", f"## Agreement with stored run `{ref}` (reported, not gated)", ""]
        for code, row in per_condition.items():
            lines.append(
                f"- {code}: decision identical {row['decision_identical']}/{row['n_shared']}, "
                f"raw identical {row['raw_identical']}/{row['n_shared']}"
            )
    if report["failures"]:
        lines += ["", "## Gate failures", ""] + [f"- {f}" for f in report["failures"]]
    if report["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = {}
    if args.model_config is not None:
        config = yaml.safe_load(args.model_config.read_text(encoding="utf-8")) or {}
    report = evaluate(args.experiment_dir, args.probe_root, config, args.gpu_mem_log)
    report["reference_agreement"] = {
        str(ref): reference_agreement(args.experiment_dir, ref) for ref in args.reference_experiment_dir
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qualification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "qualification_report.md").write_text(render_markdown(report), encoding="utf-8")
    (args.output_dir / "QUALIFICATION_VERDICT.txt").write_text(report["verdict"] + "\n", encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
