"""Visitor blind-review sessions.

Mirrors the project's semantic audit (src/semantic_review/): the visitor sees only the raw
patch and the target YOLO box, answers "Does the highlighted YOLO detection correspond to a
real palm?" with palm / non_palm / ambiguous (or skips), and only after every item is
answered are the research label, LabelMe geometry and VLM decisions revealed.

Blindness: items are addressed by random per-session tokens; neither the detection ID, the
patch name, YOLO confidence, IoU nor any label is sent before the reveal.

Visitor labels are kept in memory and appended to demo/data/visitor_reviews/*.jsonl.
Official annotation files are only ever read.
"""

from __future__ import annotations

import json
import random
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.data import loaders as L
from app.data import paths as P
from app.services import model_service

VISITOR_LABELS = ("palm", "non_palm", "ambiguous", "skip")
QUESTION = "Does the highlighted YOLO detection correspond to a real palm?"
LABEL_DEFINITIONS = {
    "palm": ("The highlighted target corresponds to a real palm crown. This includes partially visible, "
             "border-truncated or tile-cut palms, poorly placed boxes, boxes with a lot of surrounding "
             "vegetation, boxes showing only a recognizable portion of a palm, and repeated boxes on the same palm."),
    "non_palm": "The highlighted target clearly does not correspond to a real palm crown.",
    "ambiguous": "The visual evidence is insufficient to decide reliably. Do not force these into NON-PALM.",
}
DECKS = {
    "unmatched": {
        "title": "IoU-unmatched cohort",
        "description": ("Uniform random sample from the 638 detections of the 5,747-detection cohort "
                        "(YOLO confidence >= 0.5) that did not match a LabelMe palm box at IoU >= 0.5."),
        "source": "outputs/semantic_gt_review/human_review.csv",
    },
    "pilot": {
        "title": "Lower-confidence pilot",
        "description": ("Uniform random sample from the 400-item blind pilot of YOLO detections with "
                        "confidence 0.10-0.50. Outside the 5,747 cohort; no VLM predictions exist."),
        "source": "outputs/semantic_gt_review/human_confidence_pilot.csv",
    },
}
MAX_ITEMS = 20
MAX_SESSIONS = 1000

_sessions: Dict[str, Dict[str, Any]] = {}
_tokens: Dict[str, tuple] = {}
_lock = threading.Lock()


def _parse_bbox(text: str) -> List[float]:
    return [float(v) for v in json.loads(text)]


def _deck_rows(deck: str) -> pd.DataFrame:
    if deck == "unmatched":
        df = L.read_csv(P.HUMAN_REVIEW_CSV, dtype={"sample_id": str, "image_id": str})
        return df.rename(columns={"sample_id": "ref_id"})
    if deck == "pilot":
        df = L.read_csv(P.PILOT_CSV, dtype={"pilot_id": str, "image_id": str})
        return df.rename(columns={"pilot_id": "ref_id"})
    raise KeyError(deck)


def _log(event: Dict[str, Any]) -> None:
    out_dir = P.demo_writable_path("visitor_reviews")
    out_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = P.demo_writable_path("visitor_reviews", f"visitor_reviews_{day}.jsonl")
    with path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decks() -> Dict[str, Any]:
    out = {}
    for key, meta in DECKS.items():
        try:
            n = len(_deck_rows(key))
        except FileNotFoundError:
            n = 0
        out[key] = {**meta, "pool_size": n}
    return {"question": QUESTION, "labels": list(VISITOR_LABELS), "definitions": LABEL_DEFINITIONS, "decks": out}


def create_session(count: int = 10, deck: str = "unmatched") -> Dict[str, Any]:
    if deck not in DECKS:
        raise KeyError(f"unknown deck {deck}")
    count = max(1, min(int(count), MAX_ITEMS))
    rows = _deck_rows(deck)
    chosen = rows.sample(n=min(count, len(rows)), random_state=random.SystemRandom().randrange(2**31))
    session_id = secrets.token_hex(12)
    items = []
    for i, r in enumerate(chosen.itertuples(index=False)):
        token = secrets.token_hex(12)
        items.append({"index": i, "token": token, "ref_id": r.ref_id, "image_id": r.image_id,
                      "bbox": _parse_bbox(r.yolo_bbox_xywh)})
    session = {"session_id": session_id, "deck": deck, "created": _now(), "items": items, "labels": {}}
    with _lock:
        if len(_sessions) >= MAX_SESSIONS:
            oldest = next(iter(_sessions))
            for it in _sessions.pop(oldest)["items"]:
                _tokens.pop(it["token"], None)
        _sessions[session_id] = session
        for it in items:
            _tokens[it["token"]] = (session_id, it["index"])
    _log({"event": "session_created", "time": session["created"], "session_id": session_id, "deck": deck,
          "items": [it["ref_id"] for it in items]})
    return public_session(session_id)


def _image_size(image_id: str) -> Dict[str, Optional[int]]:
    lm = L.labelme_palm_boxes(image_id)
    if lm.get("image_width"):
        return {"width": int(lm["image_width"]), "height": int(lm["image_height"])}
    path = P.RAW_PATCHES_ROOT / f"{image_id}.png"
    try:
        w, h = L._png_size(path)
        return {"width": w, "height": h}
    except (FileNotFoundError, ValueError):
        return {"width": None, "height": None}


def public_session(session_id: str) -> Dict[str, Any]:
    s = _sessions.get(session_id)
    if s is None:
        raise KeyError(session_id)
    items = []
    for it in s["items"]:
        size = _image_size(it["image_id"])
        items.append({
            "index": it["index"],
            "image_url": f"/api/review/image/{it['token']}",
            "bbox_xywh": it["bbox"],
            "image_width": size["width"],
            "image_height": size["height"],
            "visitor_label": s["labels"].get(it["index"]),
        })
    return {
        "session_id": session_id,
        "deck": s["deck"],
        "deck_info": DECKS[s["deck"]],
        "question": QUESTION,
        "labels": list(VISITOR_LABELS),
        "definitions": LABEL_DEFINITIONS,
        "items": items,
        "n_answered": len(s["labels"]),
        "complete": len(s["labels"]) == len(s["items"]),
    }


def image_for_token(token: str) -> Optional[Path]:
    ref = _tokens.get(token)
    if ref is None:
        return None
    session_id, index = ref
    item = _sessions[session_id]["items"][index]
    path = P.RAW_PATCHES_ROOT / f"{item['image_id']}.png"
    return path if path.is_file() else None


def submit_label(session_id: str, index: int, label: str) -> Dict[str, Any]:
    if label not in VISITOR_LABELS:
        raise ValueError(f"label must be one of {VISITOR_LABELS}")
    with _lock:
        s = _sessions.get(session_id)
        if s is None:
            raise KeyError(session_id)
        if not 0 <= index < len(s["items"]):
            raise ValueError("item index out of range")
        s["labels"][index] = label
    _log({"event": "label", "time": _now(), "session_id": session_id, "index": index,
          "ref_id": s["items"][index]["ref_id"], "label": label})
    return public_session(session_id)


def _vlm_decisions(sample_id: str) -> Dict[str, Dict[str, Optional[str]]]:
    lookup = L.decision_lookup().get(sample_id, {})
    out = {}
    for m in model_service.list_models():
        if m["n_usable_conditions"] == 0:
            continue
        out[m["model_key"]] = {c: lookup.get((m["model_key"], c)) for c in P.CONDITIONS}
    return out


def results(session_id: str) -> Dict[str, Any]:
    s = _sessions.get(session_id)
    if s is None:
        raise KeyError(session_id)
    if len(s["labels"]) < len(s["items"]):
        raise PermissionError("reveal is available only after every item is answered or skipped")
    rows = _deck_rows(s["deck"]).set_index("ref_id")
    det = L.detections() if s["deck"] == "unmatched" else None
    pilot_ref = None
    if s["deck"] == "pilot" and P.PILOT_REFERENCE_CSV.is_file():
        pilot_ref = L.read_csv(P.PILOT_REFERENCE_CSV, dtype={"pilot_id": str}).set_index("pilot_id")
    items = []
    for it in s["items"]:
        visitor = s["labels"].get(it["index"])
        research = rows.loc[it["ref_id"], "semantic_label"]
        lm = L.labelme_palm_boxes(it["image_id"])
        entry: Dict[str, Any] = {
            "index": it["index"],
            "ref_id": it["ref_id"],
            "image_id": it["image_id"],
            "image_url": f"/api/review/image/{it['token']}",
            "bbox_xywh": it["bbox"],
            **{f"image_{k}": v for k, v in _image_size(it["image_id"]).items()},
            "visitor_label": visitor,
            "research_label": research,
            "agrees": None if visitor == "skip" else visitor == research,
            "labelme_palm_boxes_xywh": lm["boxes"],
        }
        if det is not None and it["ref_id"] in det.index:
            d = det.loc[it["ref_id"]]
            entry.update({
                "detection_id": it["ref_id"],
                "yolo_confidence": float(d["confidence"]),
                "max_iou": float(d["max_iou"]),
                "labelme_matched": bool(d["matched_gt"]),
                "unmatched_reason": d["unmatched_reason_geometric"] if isinstance(
                    d["unmatched_reason_geometric"], str) else None,
                "vlm": _vlm_decisions(it["ref_id"]),
            })
        if pilot_ref is not None and it["ref_id"] in pilot_ref.index:
            p = pilot_ref.loc[it["ref_id"]]
            entry.update({
                "yolo_confidence": float(p["yolo_confidence"]),
                "confidence_bin": p["confidence_bin"],
                "max_iou": float(p["geo_max_iou_labelme"]),
                "labelme_matched": bool(p["geo_annotation_aligned_iou050_any"]),
                "vlm": None,
            })
        items.append(entry)
    answered = [i for i in items if i["visitor_label"] != "skip"]
    _log({"event": "revealed", "time": _now(), "session_id": session_id})
    return {
        "session_id": session_id,
        "deck": s["deck"],
        "deck_info": DECKS[s["deck"]],
        "items": items,
        "n_answered": len(answered),
        "n_agree": sum(1 for i in answered if i["agrees"]),
        "models": [{"model_key": m["model_key"], "display": m["display"], "family": m["family"],
                    "degenerate": m["degenerate"],
                    "usable_conditions": [c for c, v in m["conditions"].items()
                                          if v["status"] in model_service.USABLE]}
                   for m in model_service.list_models() if m["n_usable_conditions"] > 0],
    }
