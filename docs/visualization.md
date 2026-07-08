# Visualization

Publication-quality qualitative figures for the wild palm verification thesis.

## Purpose

This pipeline generates figures comparing:

- Original orthomosaic patch
- Ground-truth LabelMe annotations
- YOLO detections
- LVM (Qwen2.5-VL) verification decisions

Figures are intended for thesis and paper use.

## Inputs

The script resolves paths automatically from `src/paths.py`:

| Artifact | Location |
|----------|----------|
| Verification inference results | `outputs/verification/qwen/<experiment_id>/A1` … `A5` |
| Evaluation CSVs | `outputs/evaluation/qwen/<experiment_id>/A1` … `A5` |
| Verification dataset overlays | `outputs/verification_dataset/images/` |
| Raw patch images | `/deac/csc/yangGrp/cuij/palm/Raw_Patches/` |
| LabelMe ground truth | Same `Raw_Patches` tree |
| YOLO predictions | `outputs/full_inference/predictions_full.json` |
| Ablation input images | `outputs/verification_ablation_<N>/` |

## Outputs

```
outputs/visualization/<experiment_id>/
    overlay/
        sample_000123_overlay.png
    comparison/
        sample_000123_comparison.png
        sample_000123_ablation.png
    failure_cases/
        sample_000045_false_positive.png
        sample_000078_false_negative.png
        sample_000091_uncertain.png
```

### Figure types

1. **Overlay** — YOLO vs GT on the original patch with center/endpoints, IoU, confidence, and LVM decision.
2. **Comparison** — Five-panel side-by-side: original, overlay, overlay+GT, overlay+YOLO, verification result text.
3. **Ablation** — One row showing A1–A5 with decision, confidence, and reasoning; best F1 ablation highlighted.
4. **Failure cases** — Automatic figures for false positives, false negatives, and uncertain predictions.

## Example command

```bash
python scripts/visualization/visualize_verification.py \
    --experiment-id 20260706_2214 \
    --sample-count 50
```

Optional arguments:

```bash
python scripts/visualization/visualize_verification.py \
    --experiment-id 20260706_2214 \
    --sample-count 50 \
    --output-dir outputs/visualization/20260706_2214 \
    --primary-ablation A3 \
    --ablation-sample-id sample_000123 \
    --seed 42
```

## Example figures

| File | Description |
|------|-------------|
| `overlay/sample_XXXXXX_overlay.png` | Green = GT bbox, red = YOLO bbox, blue = center, yellow = endpoints |
| `comparison/sample_XXXXXX_comparison.png` | Side-by-side verification workflow panels |
| `comparison/sample_XXXXXX_ablation.png` | A1–A5 ablation comparison for one sample |
| `failure_cases/sample_XXXXXX_false_positive.png` | LVM Reliable but no GT match |
| `failure_cases/sample_XXXXXX_false_negative.png` | LVM Unreliable but GT matched |
| `failure_cases/sample_XXXXXX_uncertain.png` | LVM Uncertain prediction |

## Style

- Matplotlib, 300 DPI
- White background, no axes
- Consistent colors and fonts
- Bounding-box legend on overlay figures

## Module layout

```
scripts/visualization/visualize_verification.py   # CLI entry point
src/visualization/experiment_data.py              # Path resolution and sample records
src/visualization/gt_geometry.py                  # LabelMe center/endpoints
src/visualization/publication_style.py            # Colors and typography
src/visualization/drawing.py                      # Shared drawing helpers
src/visualization/figure_overlay.py
src/visualization/figure_comparison.py
src/visualization/figure_ablation.py
src/visualization/figure_failure.py
```

## Prerequisites

Run quantitative evaluation before generating figures:

```bash
python scripts/evaluate_verification_against_groundtruth.py ...
python scripts/compute_verification_metrics.py ...
```

Evaluation protocol: [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)

Ablation design: [ABLATION_STUDY.md](ABLATION_STUDY.md)
