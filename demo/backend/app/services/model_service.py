"""Model registry: which checkpoints exist and the audited status of every model x condition cell.

Sources (all read-only):
- paper/analysis/scripts/common.py      RUNS / LADDERS / CORE_PANEL (parsed statically)
- paper/analysis/outputs/audit/model_inventory.csv   per-cell audit outcome, files present
- paper/analysis/outputs/label_usage/label_usage_by_checkpoint.csv   collapse diagnostics
- paper/analysis/derived/sample_decisions.csv.gz     valid-decision counts per cell

Cell status vocabulary:
  full_clean   all 5,747 predictions present and every integrity check passed
  partial      all files present; some rows are parse errors (only valid rows exist)
  incomplete   prediction files exist but the run has not finished / not been audited as usable
  not_run      no prediction files for this condition
  not_audited  model absent from the audit inventory
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.data import loaders as L
from app.data import paths as P

USABLE = ("full_clean", "partial")


def _split(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _files_present(value: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in _split(value):
        if ":" in item:
            cond, n = item.split(":", 1)
            try:
                out[cond.strip()] = int(n)
            except ValueError:
                pass
    return out


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _valid_counts() -> Dict[tuple, int]:
    df = L.sample_table()
    return df.groupby(["model_key", "condition"]).size().to_dict()


def _label_usage() -> Dict[str, Dict[str, Any]]:
    usage = L.analysis_csv("label_usage", "label_usage_by_checkpoint.csv")
    if usage is None:
        return {}
    return {r["model_key"]: {k: _clean(v) for k, v in r.items()} for r in usage.to_dict("records")}


def _label_usage_cells() -> Dict[tuple, Dict[str, Any]]:
    cells = L.analysis_csv("label_usage", "label_usage_by_cell.csv")
    if cells is None:
        return {}
    key_col = "model_key" if "model_key" in cells.columns else None
    out = {}
    for r in cells.to_dict("records"):
        mk = r.get(key_col) if key_col else None
        out[(mk or r.get("model"), r.get("condition"))] = {k: _clean(v) for k, v in r.items()}
    return out


def list_models() -> List[Dict[str, Any]]:
    registry = L.analysis_registry()
    inventory = L.analysis_csv("audit", "model_inventory.csv")
    inv = {r["model_key"]: r for r in inventory.to_dict("records")} if inventory is not None else {}
    counts = _valid_counts()
    usage = _label_usage()
    core = set(registry.get("CORE_PANEL", []))
    ladders = registry.get("LADDERS", {})
    policy = registry.get("CONDITION_POLICY", {})

    runs = list(registry.get("RUNS", []))
    known = {r["model_key"] for r in runs}
    for mk, row in inv.items():
        if mk not in known:
            runs.append({"model_key": mk, "display": row.get("display"), "family": row.get("family"),
                         "nominal_size": row.get("nominal_size_from_name"), "nominal_b": None,
                         "rel": None, "expected": row.get("expected_status")})

    models = []
    for order, run in enumerate(runs):
        mk = run["model_key"]
        row = inv.get(mk)
        analysed = set(_split(row.get("conditions_analysed"))) if row else set()
        partial = set(_split(row.get("conditions_partial"))) if row else set()
        files = _files_present(row.get("conditions_files_present")) if row else {}
        conditions = {}
        for cond in P.CONDITIONS:
            if row is None:
                status = "not_audited"
            elif cond in analysed:
                status = "full_clean"
            elif cond in partial:
                status = "partial"
            elif files.get(cond, 0) > 0:
                status = "incomplete"
            else:
                status = "not_run"
            conditions[cond] = {
                "status": status,
                "files_present": files.get(cond),
                "n_valid": int(counts.get((mk, cond), 0)) if status in USABLE else None,
            }
        n_usable = sum(1 for c in conditions.values() if c["status"] in USABLE)
        ladder = next((name for name, keys in ladders.items() if mk in keys), None)
        u = usage.get(mk, {})
        params = _clean(row.get("parameters_billions")) if row else None
        models.append({
            "model_key": mk,
            "display": run.get("display"),
            "family": run.get("family"),
            "nominal_size": run.get("nominal_size"),
            "nominal_b": run.get("nominal_b"),
            "params_billions": params,
            "registry_order": order,
            "run_dir": f"outputs/verification/{run['rel']}" if run.get("rel") else None,
            "expected_status": run.get("expected"),
            "condition_policy": policy.get(mk),
            "in_core_panel": run.get("display") in core,
            "ladder": ladder,
            "conditions": conditions,
            "n_usable_conditions": n_usable,
            "label_usage": {
                "pattern": u.get("descriptive_pattern"),
                "labels_never_used": u.get("labels_never_used_in_any_condition"),
                "n_single_label": u.get("n_single_label"),
                "n_majority_ge_0_99": u.get("n_majority_ge_0.99"),
                "majority_share_max": u.get("majority_share_max"),
                "entropy_min": u.get("entropy_min"),
                "entropy_max": u.get("entropy_max"),
            } if u else None,
            "degenerate": _degeneracy(u),
        })
    return models


def _degeneracy(u: Dict[str, Any]) -> Optional[str]:
    """Descriptive collapse flag mirroring a10's cut-offs (majority >= 0.99); not a scientific threshold."""
    if not u:
        return None
    n_usable = u.get("n_conditions_usable") or 0
    single = u.get("n_single_label") or 0
    major = u.get("n_majority_ge_0.99") or 0
    if n_usable and single == n_usable:
        return "single_label"
    if n_usable and major == n_usable:
        return "near_collapse_all"
    if major:
        return "near_collapse_some"
    return None


def get_model(model_key: str) -> Optional[Dict[str, Any]]:
    model = next((m for m in list_models() if m["model_key"] == model_key), None)
    if model is None:
        return None
    dist = L.analysis_csv("decision_distribution_long.csv")
    if dist is not None:
        rows = dist[dist["model_key"] == model_key]
        model["decision_distribution"] = [
            {k: _clean(v) for k, v in r.items()} for r in rows.to_dict("records")
        ]
    cells = _label_usage_cells()
    model["label_usage_by_condition"] = {
        cond: cells.get((model_key, cond)) or cells.get((model["display"], cond))
        for cond in P.CONDITIONS
    }
    return model


def models_by_key() -> Dict[str, Dict[str, Any]]:
    return {m["model_key"]: m for m in list_models()}


def run_dir(model_key: str) -> Optional[str]:
    for run in L.analysis_registry().get("RUNS", []):
        if run["model_key"] == model_key:
            return run.get("rel")
    return None
