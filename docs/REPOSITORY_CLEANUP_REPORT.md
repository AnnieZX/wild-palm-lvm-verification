# Repository Cleanup Report

July 2026 — pre multi-VLM benchmark reorganization.

## Summary

The repository was reorganized to separate production entry points, pipeline preparation scripts, shared utilities, and archived experiments. **No algorithms, evaluation logic, or prompts were modified.** Compatibility shims preserve existing import paths under `src/preprocessing/` and `src/yolo/`.

---

## Files moved to archive

### `archive/scripts/`

| File | Reason |
|------|--------|
| `visualize_yolo_gt_overlap_full.py` | Experimental YOLO/GT overlap visualization |
| `analyze_verification_ablation.py` | Supplementary ablation inference summary (not evaluation metrics) |
| `experimental/` (entire folder) | Supplementary YOLO QA (`match_predictions_to_groundtruth.py`, `check_prediction_statistics.py`) |
| `run_full_inference_and_overlay.py` | Deprecated redirect → `scripts/run_full_inference.py` |
| `run_verification_inference.py` | Deprecated redirect → `scripts/run_verification.py` |
| `submit_qwen_ablation.sh` | Deprecated redirect → `jobs/submit_ablation.sh` |

### `archive/src/preprocessing/`

| File | Reason |
|------|--------|
| `patch_extractor.py` | Empty stub |

### Pre-existing archive content (unchanged)

- `archive/old_labelme_ablation/` — superseded E1–E5 × P1–P6 ablation
- `archive/prototype/` — early mock/LVM prototypes
- `archive/deprecated_scripts/` — old debug/visualization scripts
- `archive/experiments/` — sequential batch experiments
- `archive/jobs/` — old SLURM jobs

---

## Files deleted

**None.** All removed items were moved to `archive/`.

---

## Scripts reorganized (not archived)

### Production entry points (`scripts/`)

| Script | Role |
|--------|------|
| `run_full_inference.py` | YOLO full-dataset inference (renamed from `run_full_inference_and_overlay.py`) |
| `run_verification.py` | VLM verification inference (renamed from `run_verification_inference.py`) |
| `run_ablation_verification.py` | **New** — delegates to `run_verification.py` per ablation condition |
| `evaluate_detection_matching.py` | YOLO vs LabelMe detection metrics |
| `evaluate_verification_against_groundtruth.py` | Verification sample ↔ GT matching |
| `compute_verification_metrics.py` | TP/FP/FN and summary metrics |
| `visualize_verification_examples.py` | Publication visualization |
| `debug_sample_matching.py` | Single-sample GT/YOLO debug |

### Pipeline preparation (`scripts/pipeline/`)

| Script | Role |
|--------|------|
| `generate_verification_dataset.py` | Build verification dataset from YOLO detections |
| `build_verification_prompts.py` | Build default verification prompts |
| `build_ablation_verification_prompts.py` | Build A1–A5 ablation inputs |
| `check_cluster_environment.py` | Cluster environment validation |

### Jobs (`jobs/`)

| File | Role |
|------|------|
| `submit_ablation.sh` | Submit ablation SLURM job (moved from `scripts/submit_qwen_ablation.sh`) |
| `qwen_download.slurm` | Model download job |

---

## Duplicate modules consolidated

| Shared module | Replaces duplicates in |
|---------------|------------------------|
| `src/utils/labelme_paths.py` → `resolve_labelme_json()` | `evaluate_verification_against_groundtruth.py`, `debug_sample_matching.py`, `visualize_verification_examples.py` |
| `src/utils/verification_index.py` → `index_bbox_from_row()`, `find_yolo_prediction()` | `evaluate_verification_against_groundtruth.py`, `debug_sample_matching.py` |
| `src/evaluation/gt_matching.py` | Canonical greedy matching (shim at `src/yolo/gt_matching.py`) |
| `src/preprocessing/gt_palm_bboxes.py` | Canonical GT palm bbox extraction (unchanged) |

### Compatibility shims (old import paths still work)

- `src/yolo/gt_matching.py` → re-exports `src.evaluation.gt_matching`
- `src/preprocessing/verification_visualization.py` → re-exports `src.visualization.verification_visualization`
- `src/preprocessing/verification_matching_debug.py` → re-exports `src.visualization.verification_matching_debug`

---

## New structure for multi-VLM support

```
configs/
├── model.yaml          # Existing active model config
├── qwen.yaml           # Placeholder
├── internvl.yaml       # Placeholder
└── llava.yaml          # Placeholder

src/
├── models/             # Future VLM adapter registry (placeholder)
├── evaluation/         # GT matching utilities
├── visualization/      # Verification figure rendering
├── utils/              # Shared path/index helpers
├── preprocessing/      # Dataset and overlay generation
├── lvm/                # Current Qwen adapter (unchanged)
├── prompts/            # Prompt templates (unchanged)
└── yolo/               # Predictions I/O (unchanged)
```

---

## Outputs layout

Documented in `outputs/README.md` (tracked in git). Target layout:

```
outputs/
├── full_inference/
├── verification_dataset/
├── verification_results/
├── evaluation/
├── visualization/
└── logs/
```

Legacy folders (`yolo_gt_overlap_full/`, `yolo_analysis/`, `verification_ablation_100/`) may still exist on cluster disks; path constants in `src/paths.py` were **not** changed to avoid breaking existing artifacts.

---

## Potential dead code

| Item | Notes |
|------|-------|
| `src/lvm/mock_verifier.py` | Only referenced by archived prototype scripts |
| `src/preprocessing/overlay_renderer.py` | Only referenced by archived prototype scripts |
| `src/preprocessing/lvm_input_builder.py` | Used by `verification_overlay.py` (production) and archived code |
| `src/preprocessing/palm_analyzer.py` | Used by archived experimental scripts; not used by current evaluation pipeline (GT uses `gt_palm_bboxes.py`) |
| `src/preprocessing/sequential_dataset.py` | Used by `evaluate_detection_matching.py` and archived scripts |
| `configs/model.yaml` | Active; new per-model YAMLs are placeholders only |

---

## Potential unused scripts (archived)

- `archive/scripts/analyze_verification_ablation.py` — inference quality summary; superseded for thesis metrics by `compute_verification_metrics.py` but still useful for debugging parse/inference failures
- `archive/scripts/visualize_yolo_gt_overlap_full.py` — exploratory QA
- `archive/scripts/experimental/*` — early YOLO vs GT analysis

---

## Current production pipeline

```
1. YOLO inference
   python scripts/run_full_inference.py

2. Verification dataset
   python scripts/pipeline/generate_verification_dataset.py
   python scripts/pipeline/build_verification_prompts.py

3. Ablation inputs (optional)
   python scripts/pipeline/build_ablation_verification_prompts.py

4. VLM verification
   python scripts/run_verification.py
   python scripts/run_ablation_verification.py --condition A1_overlay_only

5. Detection evaluation (pre-verification baseline)
   python scripts/evaluate_detection_matching.py

6. Verification evaluation
   python scripts/evaluate_verification_against_groundtruth.py
   python scripts/compute_verification_metrics.py

7. Visualization
   python scripts/visualize_verification_examples.py
   python scripts/debug_sample_matching.py --sample-id sample_XXXXXX
```

Cluster submission:

```bash
bash jobs/submit_ablation.sh
```

Official protocol: `docs/EVALUATION_PROTOCOL.md`

---

## Needs manual review

| Item | Concern |
|------|---------|
| `jobs/qwen_verification_ablation.slurm` | Referenced by `jobs/submit_ablation.sh` but **not present** in the repository — may exist only on cluster |
| `README.md` | Still references old script names and paths; update when convenient |
| `scripts/pipeline/` vs flat `scripts/` | Pipeline prep scripts kept in subfolder because they are production-critical but not top-level entry points |
| `src/lvm/` vs `src/models/` | VLM code remains in `src/lvm/`; `src/models/` is a placeholder for future adapters |
| `src/preprocessing/palm_analyzer.py` | Legacy LabelMe group parsing; evaluation uses `gt_palm_bboxes.py` instead — keep for archived tooling |
| `compute_verification_metrics.py` binary label handling | Code excludes `Uncertain` from TP/FP/FN; `docs/EVALUATION_PROTOCOL.md` defines Uncertain as negative — align in a future pass |
| `archive/scripts/analyze_verification_ablation.py` | Useful for inference debugging; restore to `scripts/` if still needed regularly |

---

## Safety notes

- Evaluation matching (`src/evaluation/gt_matching.py`) is unchanged in behavior.
- Prompt templates under `src/prompts/` were not modified.
- Deprecated script names in `archive/scripts/` forward to new entry points for backward compatibility.
