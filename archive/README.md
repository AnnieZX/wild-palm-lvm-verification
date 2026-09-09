# Archive

This directory holds **non-primary** research artifacts that must not drive thesis/paper metrics unless explicitly restored and revalidated.

Cleanup of June 2026 (script/doc reorganization) and September 2026 (post A1-1000 cross-model audit) both deposit material here. **Nothing here was deleted.**

## Why the archive exists

- Keep the active `outputs/` tree focused on **canonical** datasets and scientifically usable runs.
- Preserve failed, superseded, smoke, and debug artifacts for reproducibility and forensic review.
- Prevent accidental citation of degenerate or incomplete results as primary evidence.

## Do not use archived files for thesis metrics

Archived experiment outputs, evaluations, and Slurm logs are **excluded from scientific analysis** by default. To reuse anything from this tree you must:

1. Restore it to its original path (or a documented new path).
2. Re-run the audit checks in `docs/EXPERIMENT_STATUS_CANONICAL.md`.
3. Explicitly justify the restoration in the paper methods.

## Canonical runs (remain under `outputs/`)

| Role | Path |
|------|------|
| Production dataset | `outputs/verification_dataset/` (5747) |
| Shared A1–A5 inputs | `outputs/verification_ablation_{10,100,1000,5747}/` |
| Qwen A1–A5 @1000 | `outputs/verification/qwen/20260706_2214/` |
| Qwen A1–A4 @5747 | `outputs/verification/qwen/20260708_0020/` (A5 partial, incomplete) |
| LLaVA A1 @1000 | `outputs/verification/llava/20260719_1734/` (degenerate all-Reliable; evidence only) |
| Gemma A1 @1000 | `outputs/verification/gemma/20260802_1702/` (degenerate all-Reliable; evidence only) |

Authoritative narrative: **`docs/EXPERIMENT_STATUS_CANONICAL.md`**.

## Layout (September 2026 audit categories)

| Directory | Contents |
|-----------|----------|
| `failed_runs/` | Crashed or empty runs (e.g. Qwen A5 `1501`, early `20260706_2130`) |
| `superseded_runs/` | Pilots, duplicate A1–A4 re-runs, legacy flat eval/results, old LVM input trees |
| `smoke_tests/` | Model/framework smoke outputs and logs |
| `debug_runs/` | Grounding/parity diagnostics, 1-sample fixtures, early logs |
| `incomplete_experiments/` | Notes for incomplete work left in place (Qwen A5 @5747) |
| `invalid_experiments/` | Notes for collapsed-model runs left in place (LLaVA/Gemma) |
| `miscellaneous_runtime_artifacts/` | Accidental empty pip redirect files, etc. |
| `deprecated_scripts/`, `prototype/`, `experiments/`, `old_docs/`, `jobs/`, `scripts/`, `src/`, `old_labelme_ablation/`, `unused_data/` | Earlier (June 2026) cleanup |

## Manifest

Every path moved in the September 2026 audit is recorded in:

**`archive/ARCHIVE_MANIFEST.csv`**

Columns: `original_path`, `archive_path`, `category`, `experiment_id`, `model`, `reason`, `date_archived`.

## Running very old archived scripts

From the **repository root** (June 2026 layout):

```bash
python archive/prototype/scripts/prepare_lvm_inputs.py
```

Prefer active entry points under `scripts/` and `scripts/pipeline/`.
