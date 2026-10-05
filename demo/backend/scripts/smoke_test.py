"""HTTP smoke test against a running demo backend (default http://127.0.0.1:8000).

Usage: python3 scripts/smoke_test.py [base_url]
Uses only the standard library. Creates review sessions (written under demo/data/ only).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def call(path: str, body=None, expect=200):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            status, raw, ctype = r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        status, raw, ctype = e.code, e.read(), e.headers.get("Content-Type", "")
    flag = "ok " if status == expect else "BAD"
    print(f"{flag} {status} {time.time() - t0:5.2f}s {path} ({len(raw)} B)")
    if status != expect:
        print("   ", raw[:300])
    return json.loads(raw) if ctype.startswith("application/json") else raw


def main() -> None:
    s = call("/api/summary")
    print("   ", {k: s[k] for k in ("n_detections", "n_images", "labelme_matched", "labelme_unmatched",
                                    "semantic_review", "n_models_all_five_usable")})
    models = call("/api/models")["models"]
    for m in models:
        print(f"    {m['display']:<18} usable={m['n_usable_conditions']} degenerate={m['degenerate']} "
              + " ".join(f"{c}:{v['status']}" for c, v in m["conditions"].items()))
    call("/api/models/qwen3_vl")
    call("/api/models/does_not_exist", expect=404)
    d = call("/api/detections?page_size=2")
    print("    total", d["total"], d["items"][0])
    print("    unmatched+ambiguous:", call("/api/detections?matched=false&semantic=ambiguous")["total"])
    print("    qwen3_vl A1!=A5:", call("/api/detections?changed_model=qwen3_vl&page_size=1")["total"])
    call("/api/detections/random?area_q=Q1")
    det = call("/api/detections/sample_000019")
    print("   ", {k: det[k] for k in ("max_iou", "labelme_matched", "semantic_label", "unmatched_reason",
                                      "labelme", "detections_in_image")})
    p = call("/api/detections/sample_000061/predictions")
    for r in p["models"]:
        if r["model_key"] in ("qwen3_vl_2b", "internvl3_5_hf_2b", "molmo2_8b", "qwen3_vl_32b", "minicpm_v4_5"):
            print(f"    {r['display']:<18}", {c: v["state"] + ":" + str(v["decision"]) for c, v in r["cells"].items()})
    r = call("/api/detections/sample_000061/reasoning?model=qwen3_vl_2b&condition=A3")
    print("    parse-error cell:", r.get("decision"), "|", str(r.get("parse_error"))[:100])
    r = call("/api/detections/sample_000001/reasoning?model=qwen3_vl&condition=A1")
    print("    reasoning:", r["decision"], "consistent:", r["consistent_with_analysis_table"])
    call("/api/detections/sample_000001/reasoning?model=qwen3_vl_32b&condition=A1")
    pr = call("/api/detections/sample_000001/prompt/A3")
    print("   ", pr["metadata_section"][:120].replace("\n", " | "))
    for cond in ("A1", "A4", "A5"):
        call(f"/api/images/condition/{cond}/sample_000001")
    call("/api/images/raw/100_0003_0001_1")
    call("/api/images/raw/..%2Fetc", expect=404)
    cs = call("/api/analysis/context-shift")
    print("    context-shift models:", len(cs["models"]))
    sc = call("/api/analysis/scaling")
    print("    ladders:", [(lad["ladder"], [c["model"] for c in lad["checkpoints"]]) for lad in sc["ladders"]])
    df = call("/api/analysis/difficulty")
    print("    difficulty models:", len(df["models"]))
    se = call("/api/analysis/semantic-review")
    print("   ", se["unmatched_review"]["label_counts"])
    call("/api/detections/curated")

    sess = call("/api/review/sessions", {"count": 3})
    sid = sess["session_id"]
    leaked = [k for k in json.dumps(sess).split('"') if k.startswith("sample_") or k in ("palm_label", "semantic_label")]
    print("    blind payload leaks:", leaked or "none")
    call(f"/api/review/sessions/{sid}/results", expect=403)
    call(sess["items"][0]["image_url"])
    for i, lab in enumerate(("palm", "ambiguous", "skip")):
        call(f"/api/review/sessions/{sid}/labels", {"index": i, "label": lab})
    call(f"/api/review/sessions/{sid}/labels", {"index": 0, "label": "Reliable"}, expect=422)
    res = call(f"/api/review/sessions/{sid}/results")
    print("    reveal:", [(i["ref_id"], i["visitor_label"], i["research_label"], i["agrees"]) for i in res["items"]])
    ps = call("/api/review/sessions", {"count": 2, "deck": "pilot"})
    print("    pilot deck items:", len(ps["items"]))


if __name__ == "__main__":
    main()
