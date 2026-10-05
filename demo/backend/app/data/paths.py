"""Locations of the research artifacts the demo reads, and the only directory it writes.

Every research path is opened read-only. The single writable location is DEMO_DATA_DIR
(demo/data/), used for visitor review sessions; `demo_writable_path` enforces that.
"""

from __future__ import annotations

import os
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[3]
REPO = Path(os.environ.get("DEMO_REPO_ROOT", DEMO_ROOT.parent)).resolve()

OUTPUTS = REPO / "outputs"
ANALYSIS = REPO / "paper" / "analysis"
ANALYSIS_OUT = ANALYSIS / "outputs"
ANALYSIS_SCRIPTS = ANALYSIS / "scripts"
COMMON_PY = ANALYSIS_SCRIPTS / "common.py"
SAMPLE_TABLE = ANALYSIS / "derived" / "sample_decisions.csv.gz"

INDEX_CSV = OUTPUTS / "verification_dataset" / "index.csv"
DATASET_DIR = OUTPUTS / "verification_dataset"
ABLATION_ROOT = OUTPUTS / "verification_ablation_5747"
VERIFICATION_ROOT = OUTPUTS / "verification"

HUMAN_REVIEW_CSV = OUTPUTS / "semantic_gt_review" / "human_review.csv"
PILOT_CSV = OUTPUTS / "semantic_gt_review" / "human_confidence_pilot.csv"
PILOT_REFERENCE_CSV = OUTPUTS / "semantic_candidate_audit" / "semantic_pilot_reference.csv"

RAW_PATCHES_ROOT = Path(os.environ.get("DEMO_RAW_PATCHES_ROOT", "/deac/csc/yangGrp/cuij/palm/Raw_Patches"))

DEMO_DATA_DIR = Path(os.environ.get("DEMO_DATA_DIR", DEMO_ROOT / "data")).resolve()

CONDITION_DIRS = {
    "A1": "A1_overlay_only",
    "A2": "A2_overlay_confidence",
    "A3": "A3_overlay_confidence_geometry",
    "A4": "A4_overlay_crop_confidence",
    "A5": "A5_crop_only",
}
CONDITIONS = tuple(CONDITION_DIRS)
LABELS = ("Reliable", "Uncertain", "Unreliable")


def demo_writable_path(*parts: str) -> Path:
    """Resolve a path under demo/data/ and refuse anything that escapes it."""
    path = DEMO_DATA_DIR.joinpath(*parts).resolve()
    if DEMO_DATA_DIR not in path.parents and path != DEMO_DATA_DIR:
        raise PermissionError(f"refusing to write outside {DEMO_DATA_DIR}: {path}")
    if REPO / "outputs" in path.parents or REPO / "paper" in path.parents:
        raise PermissionError(f"refusing to write into research artifacts: {path}")
    return path
