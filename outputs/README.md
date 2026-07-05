# Generated artifacts (gitignored except this file)

All experiment outputs live under `outputs/`. Do not delete existing data when reorganizing; new runs should follow this layout.

```
outputs/
├── full_inference/          # YOLO predictions + overlays
│   ├── predictions_full.json
│   └── overlays/
├── verification_dataset/    # One sample per YOLO detection
│   ├── images/
│   ├── metadata/
│   ├── prompts/
│   └── index.csv
├── verification_results/    # Default VLM verification inference
├── verification_ablation_100/   # Ablation inputs (legacy path name)
├── verification_ablation_results/  # Ablation inference results
├── evaluation/              # Detection + verification metrics
├── visualization/           # Publication figures
└── logs/                    # Optional run logs (also see repo logs/)
```

Legacy folders from earlier experiments (for example `yolo_gt_overlap_full/`, `yolo_analysis/`) may still exist on disk. They are not part of the current production pipeline.

See `docs/EVALUATION_PROTOCOL.md` for the official evaluation protocol.
