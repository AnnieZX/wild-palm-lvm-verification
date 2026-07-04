# Experimental Scripts

Scripts here are supplementary analysis tools. They are not required for the main verification pipeline.

Run from the **repository root**:

```bash
python scripts/experimental/<script>.py
```

| Script | Purpose |
|--------|---------|
| `match_predictions_to_groundtruth.py` | Per-GT-palm IoU matching CSV |
| `check_prediction_statistics.py` | GT vs YOLO count comparison per image |

All scripts use shared paths from `src/paths.py` and YOLO utilities from `src/yolo/predictions_io.py`.
