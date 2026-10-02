#!/usr/bin/env python3
"""
Download one Hugging Face checkpoint at a pinned revision and verify completeness.

Refuses to write into a directory that already holds a different repo or
revision, so parameter sizes never share a checkpoint directory.

Completeness check: every file listed upstream at the pinned revision exists
locally with the upstream byte size; every shard referenced by a
``*.safetensors.index.json`` exists.

Usage:
    python scripts/pipeline/download_pinned_model.py \
        --repo-id Qwen/Qwen3-VL-4B-Instruct \
        --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
        --local-dir /deac/csc/yangGrp/luoz23/models/Qwen3-VL-4B-Instruct
    python scripts/pipeline/download_pinned_model.py ... --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

META_FILENAME = "DOWNLOAD_META.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True, help="Full 40-char commit SHA")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def read_meta(local_dir: Path) -> dict[str, str]:
    meta_path = local_dir / META_FILENAME
    if not meta_path.exists():
        return {}
    meta: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def upstream_files(repo_id: str, revision: str) -> dict[str, int | None]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Upstream sha {info.sha} != pinned revision {revision}")
    return {s.rfilename: s.size for s in info.siblings}


def verify(local_dir: Path, files: dict[str, int | None]) -> list[str]:
    problems: list[str] = []
    for name, size in sorted(files.items()):
        path = local_dir / name
        if not path.exists():
            problems.append(f"missing: {name}")
        elif size is not None and path.stat().st_size != size:
            problems.append(f"size mismatch: {name} local={path.stat().st_size} upstream={size}")
    for index_path in local_dir.glob("*.safetensors.index.json"):
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        for shard in sorted(set(weight_map.values())):
            if not (local_dir / shard).exists():
                problems.append(f"missing shard referenced by {index_path.name}: {shard}")
    return problems


def main() -> int:
    args = parse_args()
    if len(args.revision) != 40:
        print(f"ERROR: --revision must be a full commit SHA, got {args.revision!r}", file=sys.stderr)
        return 2
    local_dir: Path = args.local_dir

    meta = read_meta(local_dir)
    if meta and (meta.get("repo_id") != args.repo_id or meta.get("revision") != args.revision):
        print(
            f"ERROR: {local_dir} already holds repo_id={meta.get('repo_id')} "
            f"revision={meta.get('revision')}; refusing to overwrite.",
            file=sys.stderr,
        )
        return 2

    files = upstream_files(args.repo_id, args.revision)
    total = sum(size or 0 for size in files.values())
    print(f"repo_id={args.repo_id} revision={args.revision} files={len(files)} bytes={total}")

    if not args.verify_only:
        from huggingface_hub import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=str(local_dir))

    problems = verify(local_dir, files)
    if problems:
        print("DOWNLOAD_INCOMPLETE")
        for problem in problems:
            print(f"  {problem}")
        return 1

    if not args.verify_only:
        (local_dir / META_FILENAME).write_text(
            f"repo_id={args.repo_id}\n"
            f"revision={args.revision}\n"
            f"local_dir={local_dir}\n"
            f"files={len(files)}\n"
            f"total_bytes={total}\n"
            f"verified_utc={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    print(f"DOWNLOAD_VERIFIED {local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
