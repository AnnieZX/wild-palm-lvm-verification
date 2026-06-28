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
| `unused_data/` | Data not referenced by active code |

## Running archived scripts

From the **repository root**:

```bash
python archive/prototype/scripts/prepare_lvm_inputs.py
python archive/experiments/scripts/prepare_lvm_inputs_100_sequential.py
python archive/deprecated_scripts/debug_single_match.py
```

Archived scripts set `PROJECT_ROOT` to the repo root (not `archive/`).

## Active pipeline (not archived)

- `scripts/prepare_ablation_inputs_100.py`
- `scripts/run_qwen_ablation_100.py`
- `scripts/run_full_inference_and_overlay.py`
- `scripts/visualize_yolo_gt_overlap_full.py`
- `src/lvm/`, `src/preprocessing/`, `src/prompts/`
- `jobs/` (SLURM jobs unchanged)

## Stale SLURM references

These jobs still point at old `scripts/` paths and need path updates if re-run:

- `jobs/qwen_batch_sample.slurm` → use `archive/prototype/scripts/`
- `jobs/qwen_100_sequential.slurm` → use `archive/experiments/scripts/`

## Deleted (empty only)

- `scripts/run_patch_extraction.py` (never implemented)

## Left in place (protected / empty stubs)

- `src/preprocessing/patch_extractor.py` (empty stub; `src/preprocessing/` not modified)
- `src/prompts/__init__.py` (empty; `src/prompts/` not modified)
