# Generated artifacts (gitignored except this file)

All experiment outputs live under `outputs/`. Do not delete existing data when reorganizing; new runs should follow this layout.

```
outputs/
├── full_inference/              # YOLO predictions + overlays
│   ├── predictions_full.json
│   └── overlays/
├── verification_dataset/        # One sample per YOLO detection
│   ├── images/
│   ├── metadata/
│   ├── prompts/
│   └── index.csv
├── verification_ablation_<N>/   # A1–A5 ablation inputs (images + prompt_index.csv)
├── verification/
│   └── <model_key>/             # e.g. qwen2_5_vl, llava
│       └── <experiment_id>/
│           ├── A1/sample_*.json
│           ├── A1/results_index.csv
│           └── … A5/
├── evaluation/
│   └── <model_key>/
│       └── <experiment_id>/
│           ├── A1/A1_evaluation.csv
│           ├── A1/A1_metrics.json
│           └── … A5/
└── visualization/
    └── <model_key>/
        └── <experiment_id>/
            ├── overlay/
            ├── comparison/
            └── failure_cases/
```

## Legacy paths

Pre-freeze Qwen2.5 experiments may exist under:

```
outputs/verification/qwen/<experiment_id>/
outputs/evaluation/qwen/<experiment_id>/
```

Path helpers in `src/paths.py` detect these automatically for resume and visualization.

Older folders (`verification_results/`, `verification_ablation_results/`, `yolo_gt_overlap_full/`) may still exist on disk from earlier experiments. They are not part of the current production pipeline.

See [`docs/EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md) and [`docs/FRAMEWORK_FREEZE.md`](../docs/FRAMEWORK_FREEZE.md).
