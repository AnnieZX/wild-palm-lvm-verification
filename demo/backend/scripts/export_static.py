"""Export a static, read-only snapshot of the demo API for hosting without the backend.

Every JSON file is produced by the same service functions that back the live /api routes,
so the values are identical to the live demo; only the detection subset and image encoding
differ. Research files are only read. Output goes to demo/web/public/snapshot/ (gitignored)
and is bundled by `VITE_STATIC=1 vite build`.

    cd demo/backend && PYTHONPATH=.:.. python3 scripts/export_static.py [--n 300] [--max-side 720]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data import loaders as L  # noqa: E402
from app.data import paths as P  # noqa: E402
from app.services import (  # noqa: E402
    analysis_service as A,
    detection_service as D,
    image_service as I,
    model_service as M,
    review_service as R,
)

OUT = Path(__file__).resolve().parents[2] / "web" / "public" / "snapshot"
BASE = "snapshot"


class Images:
    """Content-addressed WebP store; identical inputs (A1-A3 share one file) are written once."""

    def __init__(self, root: Path, max_side: int, quality: int) -> None:
        self.root, self.max_side, self.quality = root, max_side, quality
        self.root.mkdir(parents=True, exist_ok=True)
        self.by_path: Dict[str, str] = {}
        self.bytes = 0

    def put(self, path: Optional[Path], max_side: Optional[int] = None) -> Optional[str]:
        if path is None:
            return None
        key = f"{path.resolve()}|{max_side}"
        if key in self.by_path:
            return self.by_path[key]
        with Image.open(path) as im:
            im = im.convert("RGB")
            side = max_side or self.max_side
            wide = im.size[0] >= 2 * im.size[1]
            im.thumbnail((side * 2, side) if wide else (side, side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="WEBP", quality=self.quality, method=5)
        data = buf.getvalue()
        name = hashlib.sha1(data).hexdigest()[:20] + ".webp"
        target = self.root / name
        if not target.exists():
            target.write_bytes(data)
            self.bytes += len(data)
        url = f"{BASE}/img/{name}"
        self.by_path[key] = url
        return url


_PRIVATE_PATH = re.compile(r"^/(deac|home|scratch|tmp)/")


def scrub(obj: Any) -> Any:
    """Publication hygiene: drop absolute cluster paths (keeping the basename, e.g. a model
    checkpoint name) and reviewer identities. Research values are not touched."""
    if isinstance(obj, dict):
        out = {k: scrub(v) for k, v in obj.items()}
        if isinstance(out.get("reviewers"), list):
            n = len(out["reviewers"])
            out["reviewers"] = [f"{n} reviewer" + ("" if n == 1 else "s")]
        return out
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str) and _PRIVATE_PATH.match(obj):
        return obj.rstrip("/").rsplit("/", 1)[-1]
    return obj


def dump(rel: str, obj: Any) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrub(obj), separators=(",", ":"), default=str))


def scrub_existing() -> None:
    n = 0
    for path in OUT.rglob("*.json"):
        raw = json.loads(path.read_text())
        clean = scrub(raw)
        if clean != raw:
            path.write_text(json.dumps(clean, separators=(",", ":"), default=str))
            n += 1
    print(f"scrubbed {n} files in {OUT}")


def pair_file(pair: str) -> str:
    return pair.replace("->", "-")


def choose_detections(n: int, review_ids: List[str], seed: int) -> List[str]:
    det = L.detections()
    curated = [c["sample_id"] for c in A.curated_cases().get("cases", [])]
    ambiguous = det.index[det["semantic_label"] == "ambiguous"].tolist()
    chosen: List[str] = []
    for sid in curated + review_ids + ambiguous:
        if sid in det.index and sid not in chosen:
            chosen.append(sid)
    rest = [s for s in det.index if s not in set(chosen)]
    random.Random(seed).shuffle(rest)
    chosen += rest[: max(0, n - len(chosen))]
    order = {s: i for i, s in enumerate(det.index)}
    return sorted(chosen, key=order.__getitem__)


def strip_reasoning(r: Dict[str, Any]) -> Dict[str, Any]:
    if r.get("available") and not r.get("parse_error") and not r.get("inference_error"):
        r = {**r, "raw_response": "", "raw_response_truncated": False, "raw_response_omitted": True}
    return r


def export_detection(sid: str, images: Images, models: List[Dict[str, Any]]) -> Dict[str, Any]:
    det = D.get_detection(sid)
    det["raw_image_url"] = images.put(I.raw_patch_path(det["image_id"]))
    for cond in P.CONDITIONS:
        det["condition_inputs"][cond] = {"image_url": images.put(I.condition_image_path(cond, sid)),
                                         "prompt_url": None}
    prompts = {c: D.get_prompt(sid, c) for c in P.CONDITIONS}
    reasoning: Dict[str, Dict[str, Any]] = {}
    for m in models:
        cells = {}
        for c in P.CONDITIONS:
            if m["conditions"][c]["status"] in M.USABLE:
                try:
                    cells[c] = strip_reasoning(D.get_reasoning(sid, m["model_key"], c))
                except KeyError as exc:
                    cells[c] = {"available": False, "reason": str(exc)}
        if cells:
            reasoning[m["model_key"]] = cells
    dump(f"detections/{sid}.json", {"detail": det, "predictions": D.get_predictions(sid), "prompts": prompts})
    dump(f"reasoning/{sid}.json", reasoning)
    return det


def export_review(deck: str, n: int, seed: int, images: Images, raw_urls: Dict[str, str]) -> List[str]:
    """Pool of blind items (image + box only) and a separate reveal file keyed by opaque item keys."""
    rows = R._deck_rows(deck)
    pool_rows = rows.sample(n=min(n, len(rows)), random_state=seed)
    det = L.detections() if deck == "unmatched" else None
    pilot_ref = None
    if deck == "pilot" and P.PILOT_REFERENCE_CSV.is_file():
        pilot_ref = L.read_csv(P.PILOT_REFERENCE_CSV, dtype={"pilot_id": str}).set_index("pilot_id")
    rng = random.Random(seed + 1)
    pool, reveal = [], {}
    for r in pool_rows.itertuples(index=False):
        key = "%016x" % rng.getrandbits(64)
        url = raw_urls.get(r.image_id) or images.put(P.RAW_PATCHES_ROOT / f"{r.image_id}.png")
        size = R._image_size(r.image_id)
        bbox = R._parse_bbox(r.yolo_bbox_xywh)
        pool.append({"key": key, "image_url": url, "bbox_xywh": bbox,
                     "image_width": size["width"], "image_height": size["height"]})
        lm = L.labelme_palm_boxes(r.image_id)
        entry: Dict[str, Any] = {"ref_id": r.ref_id, "image_id": r.image_id,
                                 "research_label": r.semantic_label, "labelme_palm_boxes_xywh": lm["boxes"]}
        if det is not None and r.ref_id in det.index:
            d = det.loc[r.ref_id]
            entry.update({
                "detection_id": r.ref_id,
                "yolo_confidence": float(d["confidence"]),
                "max_iou": float(d["max_iou"]),
                "labelme_matched": bool(d["matched_gt"]),
                "unmatched_reason": d["unmatched_reason_geometric"]
                if isinstance(d["unmatched_reason_geometric"], str) else None,
                "vlm": R._vlm_decisions(r.ref_id),
            })
        if pilot_ref is not None and r.ref_id in pilot_ref.index:
            p = pilot_ref.loc[r.ref_id]
            entry.update({
                "yolo_confidence": float(p["yolo_confidence"]),
                "confidence_bin": p["confidence_bin"],
                "max_iou": float(p["geo_max_iou_labelme"]),
                "labelme_matched": bool(p["geo_annotation_aligned_iou050_any"]),
                "vlm": None,
            })
        reveal[key] = entry
    rng.shuffle(pool)
    dump(f"review/{deck}/pool.json", pool)
    dump(f"review/{deck}/reveal.json", reveal)
    return [e["detection_id"] for e in reveal.values() if e.get("detection_id")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300, help="detections in the explorer subset")
    ap.add_argument("--review-unmatched", type=int, default=100)
    ap.add_argument("--review-pilot", type=int, default=60)
    ap.add_argument("--max-side", type=int, default=720)
    ap.add_argument("--quality", type=int, default=74)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--scrub-existing", action="store_true",
                    help="only re-apply publication scrubbing to an existing snapshot")
    args = ap.parse_args()
    if args.scrub_existing:
        scrub_existing()
        return

    if OUT.exists():
        shutil.rmtree(OUT)
    images = Images(OUT / "img", args.max_side, args.quality)
    models = M.list_models()

    review_unmatched = R._deck_rows("unmatched").sample(
        n=args.review_unmatched, random_state=args.seed)["ref_id"].tolist()
    subset = choose_detections(args.n, review_unmatched, args.seed)

    det = L.detections()
    lookup = L.decision_lookup()
    index = []
    raw_urls: Dict[str, str] = {}
    for i, sid in enumerate(subset, 1):
        d = export_detection(sid, images, models)
        raw_urls[d["image_id"]] = d["raw_image_url"]
        row = D._summary(det.loc[sid])
        dec = lookup.get(sid, {})
        row["a1_a5_changed_models"] = [
            m["model_key"] for m in models
            if dec.get((m["model_key"], "A1")) and dec.get((m["model_key"], "A5"))
            and dec[(m["model_key"], "A1")] != dec[(m["model_key"], "A5")]
        ]
        index.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(subset)} detections, images {images.bytes / 1e6:.1f} MB", flush=True)
    dump("detections/index.json", index)

    export_review("unmatched", args.review_unmatched, args.seed, images, raw_urls)
    export_review("pilot", args.review_pilot, args.seed, images, raw_urls)
    decks = R.decks()
    pool_sizes = {"unmatched": args.review_unmatched, "pilot": args.review_pilot}
    for key, info in decks["decks"].items():
        info["full_pool_size"] = info.get("pool_size")
        info["pool_size"] = pool_sizes[key]
    dump("review/decks.json", decks)
    dump("review/models.json", [
        {"model_key": m["model_key"], "display": m["display"], "family": m["family"],
         "degenerate": m["degenerate"],
         "usable_conditions": [c for c, v in m["conditions"].items() if v["status"] in M.USABLE]}
        for m in models if m["n_usable_conditions"] > 0])

    dump("summary.json", A.summary())
    dump("models.json", {"models": models})
    dump("curated.json", A.curated_cases())
    dump("analysis/scaling.json", A.scaling())
    dump("analysis/semantic-review.json", A.semantic_review())
    for pair in sorted({p["pair"] for p in A.condition_pairs()} | {"A1->A5"}):
        dump(f"analysis/context-shift/{pair_file(pair)}.json", A.context_shift(pair))
    diff = A.difficulty("A1->A5")
    for pair in diff.get("pairs", ["A1->A5"]):
        dump(f"analysis/difficulty/{pair_file(pair)}.json", A.difficulty(pair))

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    n_files = sum(1 for f in OUT.rglob("*") if f.is_file())
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_detections_total": int(len(det)),
        "n_detections_included": len(subset),
        "review_pool": pool_sizes,
        "image_encoding": f"WebP q{args.quality}, longest side <= {args.max_side}px (A4 <= {2 * args.max_side}px)",
        "seed": args.seed,
    }
    dump("meta.json", meta)
    print(json.dumps({**meta, "files": n_files, "megabytes": round(total / 1e6, 1)}, indent=2))


if __name__ == "__main__":
    main()
