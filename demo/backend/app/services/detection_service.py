"""Per-detection views: metadata, LabelMe geometry, VLM decisions, reasoning and executed prompts."""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from app.data import loaders as L
from app.data import paths as P
from app.services import model_service

SEMANTIC_FILTERS = ("palm", "ambiguous", "non_palm", "not_reviewed")


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _bbox(row: pd.Series) -> List[float]:
    return [float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_width"]), float(row["bbox_height"])]


def _summary(row: pd.Series) -> Dict[str, Any]:
    matched = _clean(row["matched_gt"])
    return {
        "detection_id": row["sample_id"],
        "image_id": row["image_name"],
        "bbox_xywh": _bbox(row),
        "bbox_area": float(row["bbox_area"]),
        "norm_area": float(row["norm_area"]),
        "image_width": int(row["image_width"]),
        "image_height": int(row["image_height"]),
        "yolo_confidence": float(row["confidence"]),
        "confidence_quintile": row["confidence_q"],
        "area_quintile": row["area_q"],
        "labelme_matched": None if matched is None else bool(matched),
        "max_iou": _clean(row["max_iou"]),
        "semantic_label": _clean(row["semantic_label"]),
        "unmatched_reason": _clean(row["unmatched_reason_geometric"]),
    }


def _filtered(
    q: Optional[str] = None,
    matched: Optional[bool] = None,
    semantic: Optional[str] = None,
    conf_min: Optional[float] = None,
    conf_max: Optional[float] = None,
    area_q: Optional[str] = None,
    conf_q: Optional[str] = None,
    changed_model: Optional[str] = None,
) -> pd.DataFrame:
    df = L.detections()
    if q:
        needle = q.strip().lower()
        if needle.isdigit():
            needle = f"sample_{int(needle):06d}"
        df = df[df["sample_id"].str.lower().str.contains(needle, regex=False)
                | df["image_name"].str.lower().str.contains(needle, regex=False)]
    if matched is not None:
        df = df[df["matched_gt"] == matched]
    if semantic:
        if semantic == "not_reviewed":
            df = df[df["semantic_label"].isna()]
        else:
            df = df[df["semantic_label"] == semantic]
    if conf_min is not None:
        df = df[df["confidence"] >= conf_min]
    if conf_max is not None:
        df = df[df["confidence"] <= conf_max]
    if area_q:
        df = df[df["area_q"] == area_q]
    if conf_q:
        df = df[df["confidence_q"] == conf_q]
    if changed_model:
        lookup = L.decision_lookup()
        keep = [
            sid for sid in df["sample_id"]
            if (a := lookup.get(sid, {}).get((changed_model, "A1"))) is not None
            and (b := lookup.get(sid, {}).get((changed_model, "A5"))) is not None and a != b
        ]
        df = df[df["sample_id"].isin(keep)]
    return df


def list_detections(page: int = 1, page_size: int = 50, **filters: Any) -> Dict[str, Any]:
    df = _filtered(**filters)
    total = len(df)
    start = max(0, (page - 1) * page_size)
    rows = df.iloc[start:start + page_size]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_summary(r) for _, r in rows.iterrows()],
    }


def random_detection(**filters: Any) -> Optional[str]:
    df = _filtered(**filters)
    if df.empty:
        return None
    return random.choice(df["sample_id"].tolist())


def get_detection(detection_id: str) -> Optional[Dict[str, Any]]:
    df = L.detections()
    if detection_id not in df.index:
        return None
    row = df.loc[detection_id]
    out = _summary(row)
    out["detections_in_image"] = int(row["detections_in_image"])
    siblings = df[df["image_name"] == row["image_name"]]
    out["other_detections"] = [
        {"detection_id": r["sample_id"], "bbox_xywh": _bbox(r), "yolo_confidence": float(r["confidence"])}
        for _, r in siblings.iterrows() if r["sample_id"] != detection_id
    ]
    lm = L.labelme_palm_boxes(row["image_name"])
    out["labelme"] = {"available": lm["available"], "palm_boxes_xywh": lm["boxes"]}
    out["condition_inputs"] = {
        cond: {"image_url": f"/api/images/condition/{cond}/{detection_id}",
               "prompt_url": f"/api/detections/{detection_id}/prompt/{cond}"}
        for cond in P.CONDITIONS
    }
    out["raw_image_url"] = f"/api/images/raw/{row['image_name']}"
    return out


def get_predictions(detection_id: str) -> Optional[Dict[str, Any]]:
    if detection_id not in L.detections().index:
        return None
    decisions = L.decision_lookup().get(detection_id, {})
    rows = []
    for model in model_service.list_models():
        mk = model["model_key"]
        cells = {}
        for cond in P.CONDITIONS:
            status = model["conditions"][cond]["status"]
            decision = decisions.get((mk, cond))
            if status in model_service.USABLE:
                state = "decision" if decision else ("parse_error" if status == "partial" else "missing")
            elif status == "incomplete":
                state = "pending"
            else:
                state = status
            cells[cond] = {"state": state, "decision": decision, "cell_status": status}
        present = [cells[c]["decision"] for c in P.CONDITIONS if cells[c]["decision"]]
        rows.append({
            "model_key": mk,
            "display": model["display"],
            "family": model["family"],
            "nominal_size": model["nominal_size"],
            "params_billions": model["params_billions"],
            "degenerate": model["degenerate"],
            "cells": cells,
            "changes_across_conditions": len(set(present)) > 1,
        })
    return {"detection_id": detection_id, "conditions": list(P.CONDITIONS), "models": rows}


_SAFE_ID = re.compile(r"^sample_\d{6}$")


def get_reasoning(detection_id: str, model_key: str, condition: str) -> Dict[str, Any]:
    if not _SAFE_ID.match(detection_id) or condition not in P.CONDITIONS:
        raise KeyError("invalid id or condition")
    model = model_service.models_by_key().get(model_key)
    if model is None:
        raise KeyError(f"unknown model {model_key}")
    status = model["conditions"][condition]["status"]
    if status not in model_service.USABLE:
        return {"available": False, "reason": f"cell status {status}: predictions not audited as usable"}
    rel = model_service.run_dir(model_key)
    path = P.VERIFICATION_ROOT / str(rel) / condition / f"{detection_id}.json"
    if not path.is_file():
        return {"available": False, "reason": "prediction file not found"}
    record = json.loads(path.read_text())
    table_decision = L.decision_lookup().get(detection_id, {}).get((model_key, condition))
    raw = record.get("raw_response") or ""
    return {
        "available": True,
        "source": str(path.relative_to(P.REPO)),
        "decision": record.get("decision") or None,
        "analysis_table_decision": table_decision,
        "consistent_with_analysis_table": (record.get("decision") or None) == table_decision,
        "visual_reasoning": record.get("visual_reasoning") or "",
        "confidence_reasoning": record.get("confidence_reasoning") or "",
        "parse_error": record.get("parse_error") or "",
        "inference_error": record.get("inference_error") or "",
        "raw_response": raw[:4000],
        "raw_response_truncated": len(raw) > 4000,
        "runtime_seconds": record.get("runtime_seconds"),
        "model_name": record.get("model_name"),
        "experiment_id": record.get("experiment_id"),
    }


def _sections(text: str) -> List[str]:
    parts = re.split(r"\n-{10,}\n", text)
    return [p.strip() for p in parts if p.strip()]


def get_prompt(detection_id: str, condition: str) -> Optional[Dict[str, Any]]:
    if not _SAFE_ID.match(detection_id) or condition not in P.CONDITIONS:
        return None
    path = P.ABLATION_ROOT / P.CONDITION_DIRS[condition] / "prompts" / f"{detection_id}.txt"
    if not path.is_file():
        return None
    text = path.read_text()
    sections = _sections(text)
    input_image = next((s for s in sections if s.startswith("Input image")), None)
    metadata = next((s for s in sections if s.startswith("Detection metadata") or s.startswith("Judge only")), None)
    return {
        "condition": condition,
        "condition_dir": P.CONDITION_DIRS[condition],
        "source": str(path.relative_to(P.REPO)),
        "input_image_section": input_image,
        "metadata_section": metadata,
        "full_text": text,
    }
