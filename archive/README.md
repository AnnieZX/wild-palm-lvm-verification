# Archive

Obsolete, experimental, and superseded files moved here during repository cleanup (June 2026).
History is preserved via `git mv`.

## Layout

| Directory | Contents |
|-----------|----------|
| `prototype/scripts/` | Local 5-tile sample pipeline (LabelMe GT → mock/small Qwen) |
| `experiments/scripts/` | Intermediate 100-palm sequential LVM run (pre-ablation) |
| `deprecated_scripts/` | YOLO val-export debug and analysis utilities |
| `old_docs/` | Superseded documentation |
| `scripts/` | July 2026 cleanup: experimental visualization, YOLO QA, deprecated entry-point redirects |
| `src/preprocessing/` | Empty stubs (e.g. `patch_extractor.py`) |

## Running archived scripts

From the **repository root**:

```bash
python archive/prototype/scripts/prepare_lvm_inputs.py
python archive/experiments/scripts/prepare_lvm_inputs_100_sequential.py
python archive/deprecated_scripts/debug_single_match.py
```

Archived scripts set `PROJECT_ROOT` to the repo root (not `archive/`).

## Active pipeline (not archived)

See `docs/REPOSITORY_CLEANUP_REPORT.md` for the current layout.

Production entry points live under `scripts/` and `scripts/pipeline/`.
Deprecated names forward from `archive/scripts/` (e.g. `run_full_inference_and_overlay.py` → `scripts/run_full_inference.py`).

## Old LabelMe ablation (superseded)

The original E1–E5 × P1–P6 LabelMe ablation pipeline was removed from active scripts.
Archived under `archive/old_labelme_ablation/`.

## Stale SLURM references

These jobs were moved to `archive/jobs/` because they reference archived scripts:

| Job | Archived script target |
|-----|------------------------|
| `archive/jobs/qwen_batch_sample.slurm` | `archive/prototype/scripts/` |
| `archive/jobs/qwen_100_sequential.slurm` | `archive/experiments/scripts/` |

Old ablation jobs: `archive/old_labelme_ablation/jobs/`

## Deleted (empty only)

- `scripts/run_patch_extraction.py` (never implemented)

## Left in place (protected / empty stubs)

- `src/preprocessing/patch_extractor.py` (empty stub; `src/preprocessing/` not modified)
- `src/prompts/__init__.py` (empty; `src/prompts/` not modified)
