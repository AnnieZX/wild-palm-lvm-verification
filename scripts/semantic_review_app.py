#!/usr/bin/env python3
"""
Purpose:
    Blind human semantic-GT review web app (labels only; reads no model output).
    Question: "Does the highlighted YOLO detection correspond to a real palm?"
    Labels: palm / non_palm / ambiguous.

Modes (never mixed):
    --mode unmatched     638 Protocol v2 GT- detections
                         source: outputs/semantic_gt_evaluation/review/semantic_review_manifest.csv
                         labels: outputs/semantic_gt_review/human_review.csv
    --mode positive-qc   existing 100-row matched-positive QC sample
                         source: outputs/diagnostics/gt_positive_blind_qc/blind_manifest.csv
                         labels: outputs/semantic_gt_review/human_positive_qc.csv
    --mode confidence-pilot  400-row lower-confidence blind pilot
                         source: outputs/semantic_candidate_audit/semantic_pilot_manifest.csv
                         (the sealed semantic_pilot_reference.csv is never read)
                         labels: outputs/semantic_gt_review/human_confidence_pilot.csv

Source manifests, predictions and evaluation outputs are never written. Images are
rendered on demand from Raw_Patches and the target YOLO box only.

Example (on DEAC):
    python3 scripts/semantic_review_app.py --reviewer zixiao --port 8765
Then on the Mac:
    ssh -N -L 8765:127.0.0.1:8765 <user>@<login-node>
and open the printed http://localhost:8765/?token=... URL.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.semantic_gt import PROTOCOL_V2_REFERENCE_CSV, SEMANTIC_REVIEW_MANIFEST_CSV  # noqa: E402
from src.paths import RAW_PATCHES_ROOT  # noqa: E402
from src.semantic_review.server import ReviewApp, make_server  # noqa: E402
from src.semantic_review.store import (  # noqa: E402
    CONFIDENCE_PILOT_MANIFEST_CSV,
    DEFAULT_OUTPUT,
    MODE_CONFIDENCE_PILOT,
    MODE_UNMATCHED,
    MODES,
    POSITIVE_QC_MANIFEST_CSV,
    ReviewDataError,
    ReviewStore,
    build_items,
    check_output_path,
    load_confidence_pilot_rows,
    load_positive_qc_rows,
    load_unmatched_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind semantic-GT review web app.")
    parser.add_argument("--reviewer", help="Reviewer identifier stored with every label (required)")
    parser.add_argument("--mode", choices=MODES, default=MODE_UNMATCHED)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="Loopback only (127.0.0.1 or localhost)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Label CSV (default by mode under outputs/semantic_gt_review/: "
                             "human_review.csv, human_positive_qc.csv or human_confidence_pilot.csv)")
    parser.add_argument("--token", default=None, help="Access token (default: random per run)")
    args = parser.parse_args()
    if not args.reviewer or not args.reviewer.strip():
        parser.error("--reviewer is required, e.g. --reviewer zixiao. "
                     "It is stored with every label; no default is invented.")
    return args


def main() -> None:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUT[args.mode]
    try:
        if args.mode == MODE_UNMATCHED:
            sources = [SEMANTIC_REVIEW_MANIFEST_CSV, PROTOCOL_V2_REFERENCE_CSV]
            rows = load_unmatched_rows(SEMANTIC_REVIEW_MANIFEST_CSV, PROTOCOL_V2_REFERENCE_CSV)
        elif args.mode == MODE_CONFIDENCE_PILOT:
            sources = [CONFIDENCE_PILOT_MANIFEST_CSV]
            rows = load_confidence_pilot_rows(CONFIDENCE_PILOT_MANIFEST_CSV)
        else:
            sources = [POSITIVE_QC_MANIFEST_CSV, PROTOCOL_V2_REFERENCE_CSV]
            rows = load_positive_qc_rows(POSITIVE_QC_MANIFEST_CSV, PROTOCOL_V2_REFERENCE_CSV)
        output = check_output_path(output, sources, args.mode)
        items = build_items(args.mode, rows, RAW_PATCHES_ROOT)
        store = ReviewStore(args.mode, items, output, args.reviewer)
        app = ReviewApp(store, token=args.token)
        server = make_server(app, args.host, args.port)
    except (ReviewDataError, ValueError) as error:
        sys.exit(f"STARTUP VALIDATION FAILED (server not started):\n{error}")
    except OSError as error:
        sys.exit(f"Could not bind {args.host}:{args.port}: {error}. Try another --port.")

    print("Semantic GT review")
    print(f"  mode:      {args.mode}")
    print(f"  source:    {sources[0]}")
    print(f"  labels:    {output}")
    print(f"  log:       {store.log_path}")
    print(f"  reviewer:  {store.reviewer}")
    print(f"  items:     {len(items)} (validated: ids, images, boxes, label file identity)")
    print(f"  completed: {store.completed()} / {len(items)}")
    print(f"  node:      {socket.gethostname()}")
    print()
    print("Mac tunnel (separate terminal; replace <user> with your DEAC username):")
    print(f"  ssh -N -L {args.port}:127.0.0.1:{args.port} <user>@{socket.gethostname()}")
    print("Open in the Mac browser:")
    print(f"  http://localhost:{args.port}/?token={app.token}")
    print("Stop with Ctrl-C (every label is already saved).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Labels saved in", output)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
