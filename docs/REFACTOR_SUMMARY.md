# Refactoring Summary (July 2026)

This document records the repository cleanup performed before continuing verification pipeline work.

## New shared modules

| Module | Purpose |
|--------|---------|
| `src/paths.py` | Central data roots and `outputs/` path constants |
| `src/yolo/predictions_io.py` | YOLO JSON loading, IoU, NMS, grouping |

## Files kept in `scripts/` (active pipeline)

| Script | Role |
|--------|------|
| `prepare_ablation_inputs_100.py` | E1–E5 overlay generation |
| `run_qwen_ablation_100.py` | Full ablation inference |
| `analyze_ablation_100.py` | Ablation summarization |
| `run_full_inference_and_overlay.py` | YOLO11x full-dataset inference |
| `visualize_yolo_gt_overlap_full.py` | GT vs YOLO overlap visualization |
| `check_cluster_environment.py` | Cluster GPU sanity check |

## Files moved to `scripts/experimental/`

| From | To |
|------|-----|
| `scripts/run_qwen_ablation_smoke_test.py` | `scripts/experimental/run_qwen_ablation_smoke_test.py` |
| `scripts/match_predictions_to_groundtruth.py` | `scripts/experimental/match_predictions_to_groundtruth.py` |
| `scripts/check_prediction_statistics.py` | `scripts/experimental/check_prediction_statistics.py` |

## Files moved to `archive/jobs/`

| From | To |
|------|-----|
| `jobs/qwen_batch_sample.slurm` | `archive/jobs/qwen_batch_sample.slurm` |
| `jobs/qwen_100_sequential.slurm` | `archive/jobs/qwen_100_sequential.slurm` |

## Output path changes

| Old path | New path |
|----------|----------|
| `outputs/yolo_gt_matches.csv` | `outputs/yolo_analysis/gt_matches.csv` |
| `outputs/prediction_statistics.csv` | `outputs/yolo_analysis/prediction_statistics.csv` |

Ablation and full-inference output paths are unchanged.

## Duplication removed

Consolidated into `src/yolo/predictions_io.py`:

- `load_predictions`, `group_predictions_by_image`, `iou_xywh`
- `filter_by_score`, `apply_nms`, `find_png_images`

Added `find_labelme_json_files()` to `sequential_dataset.py`.

## Files that can probably be deleted later

After confirming nothing references them:

| Path | Reason |
|------|--------|
| `src/preprocessing/patch_extractor.py` | Empty stub |
| `src/prompts/__init__.py` | Empty file |
| `archive/deprecated_scripts/*` | Superseded YOLO debug scripts |
| `archive/prototype/scripts/*` | Local 5-tile prototype |
| `archive/experiments/scripts/*` | Pre-ablation sequential 100 run |
| `archive/unused_data/annotations.json` | Unused platform export (~381k lines) |
| `archive/jobs/qwen_batch_sample.slurm` | Points to archived prototype scripts |
| `archive/jobs/qwen_100_sequential.slurm` | Points to archived experiment scripts |

**Do not delete** until thesis reproducibility is confirmed.

## Already in `archive/` (prior cleanup)

See `archive/README.md` for prototype scripts, deprecated YOLO debug tools, and old docs.
