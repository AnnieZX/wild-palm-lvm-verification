# Experimental Scripts

Scripts here are supplementary or pre-flight checks. They are not required for the main ablation pipeline but remain available for analysis and debugging.

Run from the **repository root**:

```bash
python scripts/experimental/<script>.py
```

| Script | Purpose |
|--------|---------|
| `run_qwen_ablation_smoke_test.py` | 2 palms × 10 conditions before full ablation |
| `match_predictions_to_groundtruth.py` | Per-GT-palm IoU matching CSV |
| `check_prediction_statistics.py` | GT vs YOLO count comparison per image |

All scripts use shared paths from `src/paths.py` and YOLO utilities from `src/yolo/predictions_io.py`.
