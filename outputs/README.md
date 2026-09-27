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
├── evaluation_protocol_v2/      # CURRENT evaluation (Evaluation Protocol v2)
│   ├── PROTOCOL.json            # protocol version, GT rule, base commit, timestamp
│   ├── rescore_manifest.csv     # v1 vs v2 metrics + invariant checks per run
│   ├── detection_metrics.json   # YOLO detection-level metrics (v2 GT)
│   └── <model_key>/
│       └── <experiment_id>/
│           ├── PROTOCOL.json
│           ├── A1/A1_evaluation.csv
│           ├── A1/A1_metrics.json
│           └── … A5/
├── evaluation/                  # Protocol v1 (case-sensitive "palm"); FROZEN, do not write
│   └── <model_key>/<experiment_id>/A*/…
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
outputs/evaluation_protocol_v2/qwen/<experiment_id>/
outputs/evaluation/qwen/<experiment_id>/            # Protocol v1
```

Path helpers in `src/paths.py` detect these automatically for resume and visualization; `CURRENT_EVALUATION_ROOT` points at `evaluation_protocol_v2/`.

`outputs/evaluation/` holds Protocol v1 results and is kept unmodified as provenance. Regenerate v2 metrics from stored predictions with `python scripts/rescore_protocol_v2.py` (no inference).

Older folders (`verification_results/`, `verification_ablation_results/`, `yolo_gt_overlap_full/`) may still exist on disk from earlier experiments. They are not part of the current production pipeline.

See [`docs/EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md) and [`docs/FRAMEWORK_FREEZE.md`](../docs/FRAMEWORK_FREEZE.md).
