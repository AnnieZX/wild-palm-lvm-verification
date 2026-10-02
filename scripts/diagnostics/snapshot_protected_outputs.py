#!/usr/bin/env python3
"""
Purpose:
    Record or verify SHA-256 hashes of experiment artifacts that the semantic-GT
    evaluation path must never modify (VLM predictions, Protocol v1/v2 evaluation
    outputs, the canonical detection index, YOLO predictions, existing GT audit
    manifests). `outputs/` is gitignored, so git cannot detect such changes.

Output:
    --write PATH   JSON snapshot {relative_path: sha256}
    --verify PATH  exit 1 if any snapshotted file changed or disappeared

Example:
    python scripts/diagnostics/snapshot_protected_outputs.py \
        --write outputs/semantic_gt_evaluation/provenance/protected_outputs_baseline.json
    python scripts/diagnostics/snapshot_protected_outputs.py \
        --verify outputs/semantic_gt_evaluation/provenance/protected_outputs_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import OUTPUTS_DIR  # noqa: E402

PROTECTED_DIRS = [
    OUTPUTS_DIR / "verification",
    OUTPUTS_DIR / "evaluation_protocol_v2",
    OUTPUTS_DIR / "evaluation",
    OUTPUTS_DIR / "diagnostics" / "gt_negative_audit",
    OUTPUTS_DIR / "diagnostics" / "gt_negative_blind_pilot",
    OUTPUTS_DIR / "diagnostics" / "gt_positive_blind_qc",
]
PROTECTED_FILES = [
    OUTPUTS_DIR / "verification_dataset" / "index.csv",
    OUTPUTS_DIR / "full_inference" / "predictions_full.json",
]
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def protected_paths() -> list[Path]:
    paths = [path for path in PROTECTED_FILES if path.is_file()]
    for root in PROTECTED_DIRS:
        if root.is_dir():
            paths.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() not in SKIP_SUFFIXES
            )
    return sorted(set(paths))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_all(paths: list[Path], workers: int) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(sha256, paths))
    return {str(path.relative_to(PROJECT_ROOT)): digest for path, digest in zip(paths, digests)}


def main() -> None:
    args = parse_args()
    if args.write:
        if args.write.exists():
            sys.exit(f"Refusing to overwrite existing snapshot: {args.write}")
        hashes = hash_all(protected_paths(), args.workers)
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with args.write.open("w", encoding="utf-8") as file:
            json.dump({
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "skipped_suffixes": sorted(SKIP_SUFFIXES),
                "files": hashes,
            }, file, indent=1)
        print(f"Wrote {len(hashes)} hashes to {args.write}")
        return

    with args.verify.open(encoding="utf-8") as file:
        expected: dict[str, str] = json.load(file)["files"]
    paths = [PROJECT_ROOT / rel for rel in expected]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in paths if not p.is_file()]
    current = hash_all([p for p in paths if p.is_file()], args.workers)
    changed = [rel for rel, digest in current.items() if expected[rel] != digest]
    new_files = sorted(
        str(p.relative_to(PROJECT_ROOT)) for p in protected_paths()
        if str(p.relative_to(PROJECT_ROOT)) not in expected
    )
    print(f"Checked {len(expected)} files: {len(changed)} changed, {len(missing)} missing, "
          f"{len(new_files)} new files in protected locations")
    for label, items in (("CHANGED", changed), ("MISSING", missing), ("NEW", new_files)):
        for item in items[:50]:
            print(f"  {label}: {item}")
    if changed or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
