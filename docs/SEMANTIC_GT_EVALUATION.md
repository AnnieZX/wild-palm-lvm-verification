# Semantic-GT Evaluation

> [!CAUTION]
> **SUPERSEDED / NOT FOR CURRENT RESULTS (2026-10-01).**
> This earlier semantic-GT implementation (`src/evaluation/semantic_gt.py`,
> `scripts/evaluate_semantic_gt.py`, `scripts/build_semantic_review_manifest.py` and everything
> under `outputs/semantic_gt_evaluation/`) **predates the official review workflow** and must not be
> used for current reporting.
>
> - It reads `outputs/semantic_gt_evaluation/review/semantic_review_manifest.csv`, which was never
>   filled (0 of 638 rows labelled). It does **not** consume the official labels in
>   `outputs/semantic_gt_review/human_review.csv`.
> - It expects fields (`review_reason`, `reviewer_notes`, reason-category breakdowns) that the
>   official review never collected.
> - Its stored outputs (`semantic_metrics.csv`, `protocol_v2_vs_semantic.csv`,
>   `gt_disagreement_summary.*`, …) reflect a 0/638-reviewed state ("semantic GT incomplete").
>
> The canonical semantic result is [SEMANTIC_VALIDITY_AUDIT.md](SEMANTIC_VALIDITY_AUDIT.md)
> (638 LabelMe-unmatched: 619 palm / 19 ambiguous / 0 non-palm). The geometry-only IoU table in §7
> is unaffected by this notice. The text below is kept unchanged for provenance.

A second, independent evaluation of the stored VLM verification decisions. It **adds**
to [Evaluation Protocol v2](EVALUATION_PROTOCOL.md) and does not replace it: Protocol v2
labels, thresholds, evaluators and outputs are unchanged and remain the canonical
annotation-alignment results.

Semantic re-evaluation reuses the existing VLM inference outputs
(`outputs/verification/<model>/<experiment>/<A*>/sample_*.json`). It does not require
rerunning any model, and changes no prompt, parser or adapter.

## 1. Two GT definitions

| | Protocol v2 — annotation-alignment GT | Semantic GT — human-reviewed object-presence GT |
|---|---|---|
| Question | Is this YOLO detection matched to a LabelMe palm box? | Does this YOLO detection visually correspond to a palm? |
| Positive | Detection matched to a LabelMe palm GT box at IoU ≥ 0.5 (greedy one-to-one) | Detection visually corresponds to a palm (`palm`) |
| Negative | Detection unmatched at IoU ≥ 0.5 | Detection does not correspond to a palm (`non_palm`) |
| Other | — | `ambiguous`: cannot confidently determine; excluded from binary metrics |
| Source | LabelMe boxes + matching (`src/evaluation/gt_matching.py`) | Human review of the 638 Protocol v2 negatives; Protocol v2 positives inherited |

**"Unmatched" does not inherently mean "non-palm."** LabelMe annotations are not
documented as exhaustive, so a Protocol v2 negative can be a real palm with no LabelMe
box (missing annotation), a palm whose LabelMe box overlaps the YOLO box at IoU < 0.5
(localization mismatch), or a second detection of a palm whose box was already
matched to a better-overlapping detection (duplicate detection; 25 such rows have
IoU ≥ 0.5 with a GT box already taken). Protocol v2 stays valid as an annotation-alignment
evaluation; semantic GT measures palm correctness.

## 2. Canonical detection set and key

- 5,747 detections, `outputs/verification_dataset/index.csv` (YOLO confidence ≥ 0.5).
- Stable key: **`sample_id`** (`sample_000001` … `sample_005747`). It names the prediction
  files and is the key column of every Protocol v2 evaluation CSV. Semantic labels
  are joined by `sample_id` only.
- Protocol v2: 5,109 GT+ (matched) / 638 GT− (unmatched).

## 3. Semantic GT schema

One row per detection in `outputs/semantic_gt_evaluation/semantic_gt_table.csv`:

| Column | Meaning |
|---|---|
| `sample_id` | Detection ID (stable key) |
| `image_id`, `image_path` | LabelMe patch ID and raw patch PNG |
| `yolo_bbox_xywh`, `yolo_confidence` | YOLO box (x, y, w, h, pixels) and confidence, identical to the Protocol v2 CSVs |
| `max_iou` | IoU with the matched GT (GT+) or best IoU with any GT ignoring assignment (GT−) |
| `matched_gt_index` | Index of the matched LabelMe palm box within the image (blank for GT−) |
| `nearest_gt_index`, `nearest_gt_bbox_xywh` | Highest-IoU LabelMe palm box |
| `nearest_gt_owner_sample_id` | Detection that owns that GT box under Protocol v2 matching |
| `original_protocol_v2_gt` | `positive` / `negative` (never edited) |
| `semantic_gt` | `palm`, `non_palm`, `ambiguous`, or blank (= unreviewed) |
| `review_reason` | see below |
| `reviewer_notes` | free text |
| `label_provenance` | `inherited_positive`, `manually_reviewed_negative_pool`, `unreviewed_negative_pool` |

`review_reason` values and allowed combinations:

| `semantic_gt` | Allowed `review_reason` |
|---|---|
| `palm` (inherited) | `matched_existing_gt` (reserved; never entered manually) |
| `palm` (reviewed) | `missing_annotation`, `localization_mismatch`, `duplicate_detection`, `other` |
| `non_palm` | `true_non_palm`, `other` |
| `ambiguous` | `ambiguous`, `other` |
| blank | blank |

`duplicate_detection` is an addition to the originally proposed reason list: it
separates the palm-duplicate rows from `localization_mismatch` (the palm has a LabelMe
box that fits well, but another detection took it).

### Positive-class rule

Protocol v2 GT+ detections inherit `semantic_gt = palm`, `review_reason = matched_existing_gt`,
`label_provenance = inherited_positive`. They were **not** manually reviewed; a
detection with IoU ≥ 0.5 against a LabelMe palm box is taken to show that palm. The
638 Protocol v2 GT− detections are the manual-review population.

## 4. Human review

1. Build (once) the reference and the blank manifest:
   `python scripts/build_semantic_review_manifest.py`
   This recomputes Protocol v2 matching from YOLO + LabelMe, verifies 5,747 / 5,109 / 638,
   cross-checks every stored Protocol v2 evaluation CSV, and never overwrites an existing
   manifest.
2. Fill `outputs/semantic_gt_evaluation/review/semantic_review_manifest.csv`
   (638 rows, one per Protocol v2 GT−). Edit only `semantic_gt`, `review_reason` and
   `reviewer_notes`; leave every other column unchanged (identity columns are validated).
   The `audit_panel_path` column points to the existing panel in
   `outputs/diagnostics/gt_negative_audit/images/` (legend in that directory's README).
3. Labels come from visual review only. They are never inferred from filenames, YOLO
   confidence, IoU, VLM outputs or any model prediction. Do not look at VLM outputs
   while labelling.
4. `outputs/` is gitignored: back up the filled manifest.

The older `outputs/diagnostics/gt_negative_audit/manifest.csv` (same 638 detections, a
richer schema with `semantic`/`cause`, currently empty) is not read by this pipeline.
If labels are entered there instead, they must be transferred explicitly
(`annotation_missing` → `missing_annotation`, `false_positive` → `true_non_palm`,
`matching_artifact` → `other` with a note).

## 5. Metrics

Decision policy is **unchanged from Protocol v2** (`scripts/compute_verification_metrics.py`):

| Model decision | Role |
|---|---|
| Reliable | positive prediction |
| Unreliable | negative prediction |
| Uncertain | excluded from TP/TN/FP/FN; counted and reported separately |

Semantic GT roles: `palm` = positive, `non_palm` = negative, `ambiguous` excluded,
blank (unreviewed) excluded. **Unreviewed rows are never treated as negative.**

Accuracy, Precision, Sensitivity (Recall), Specificity, F1 and Balanced Accuracy use
the Protocol v2 formulas (the same `compute_metrics` function). In the semantic outputs,
a ratio with a zero denominator is left blank instead of 0.

### Incomplete review

Until all 638 rows are labelled, the status is **`semantic GT incomplete`**:

- no full-dataset semantic metrics are reported (the `all_5747` semantic row is blank);
- semantic metrics are computed only on the reviewed subset = inherited positives +
  reviewed negative-pool rows (excluding `ambiguous`), and the scope string states the
  denominator (e.g. `reviewed_subset (inherited positives + 120/638 reviewed …)`);
- this subset has a different class balance from the full set, and if review follows
  `priority_review.csv` it is not a random sample of the negatives. Read it as
  provisional.

### Side-by-side comparison

`protocol_v2_vs_semantic.csv` has, per model / experiment / ablation:

1. Protocol v2, `all_5747`: the stored `A*_metrics.json` values (not recomputed for display;
   reproducibility is asserted separately).
2. Protocol v2 on the semantic-evaluable rows: same rows as (3), Protocol v2 labels.
3. Semantic GT on the semantic-evaluable rows.
4. (Incomplete review only) Semantic, `all_5747`: `semantic GT incomplete`, blank metrics.

Comparing (2) with (3) isolates the effect of the GT definition at a fixed denominator;
comparing (1) with (3) also includes the removal of ambiguous / unreviewed rows.

## 6. GT disagreement analysis (the 638 Protocol v2 negatives)

`gt_disagreement_summary.csv/.json`:

| Category | Definition |
|---|---|
| A true semantic negatives | `semantic_gt = non_palm` |
| B missing annotation | `palm` + `missing_annotation` |
| C localization mismatch | `palm` + `localization_mismatch` |
| palm duplicate detection | `palm` + `duplicate_detection` |
| palm other | `palm` + `other` |
| D ambiguous | `ambiguous` |
| unreviewed | blank |

Key quantity: **semantic palm among original GT negatives / 638**, which measures how
contaminated the Protocol v2 negative class is. While review is incomplete, it is
reported only as bounds (unreviewed rows counted as all non_palm or all palm) and as a
fraction of the reviewed rows.

`negative_pool_decisions_by_semantic.csv` gives, per run, Reliable / Uncertain / Unreliable
counts within the 638 for each semantic category.

## 7. IoU sensitivity (analysis only)

`scripts/analysis/iou_sensitivity_analysis.py` recomputes the unchanged Protocol v2 matching
(same 5,747 detections, same LabelMe palm rule, greedy one-to-one) at IoU 0.30 / 0.40 /
0.50 / 0.60 / 0.70 from the existing YOLO and LabelMe files, and asserts that 0.50 reproduces
Protocol v2 exactly. A lower threshold is not "better"; the canonical threshold stays 0.5.

| IoU ≥ | matched | unmatched | positive % | negative % | v2 GT− now matched | v2 GT+ now unmatched |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 5,190 | 557 | 90.31 | 9.69 | 81 | 0 |
| 0.40 | 5,160 | 587 | 89.79 | 10.21 | 51 | 0 |
| **0.50** | **5,109** | **638** | **88.90** | **11.10** | 0 | 0 |
| 0.60 | 4,983 | 764 | 86.71 | 13.29 | 0 | 126 |
| 0.70 | 4,719 | 1,028 | 82.11 | 17.89 | 0 | 390 |

Only 81 of the 638 Protocol v2 negatives overlap a free LabelMe palm box at IoU ≥ 0.3, so
most unmatched detections are not near-misses of an existing box. Once labels exist,
`semantic_rematch_by_threshold.csv` reports how many semantic-palm negatives (by
`review_reason`) would be matched at each threshold. This helps separate localization
mismatch from missing annotation.

## 8. Commands

```bash
# once: reference + blank manifest (refuses to overwrite a manifest)
python scripts/build_semantic_review_manifest.py

# after (or during) human review
python scripts/evaluate_semantic_gt.py                    # partial review allowed, labelled as incomplete
python scripts/evaluate_semantic_gt.py --require-complete # fails unless all 638 rows are labelled
python scripts/analysis/iou_sensitivity_analysis.py

# validation
python -m unittest tests.test_semantic_gt_evaluation -v
python scripts/diagnostics/snapshot_protected_outputs.py \
    --verify outputs/semantic_gt_evaluation/provenance/protected_outputs_baseline.json
```

`evaluate_semantic_gt.py` refuses to write under `outputs/evaluation/`,
`outputs/evaluation_protocol_v2/` or `outputs/verification/`. For every run, it asserts
that the raw prediction decisions equal the Protocol v2 CSV labels, that `matched_gt`
equals the reference, and that the stored Protocol v2 metrics are reproduced exactly.

## 9. Outputs

```
outputs/semantic_gt_evaluation/
├── reference/protocol_v2_detection_reference.csv   # 5,747 rows, Protocol v2 GT
├── review/semantic_review_manifest.csv             # 638 rows, human labels
├── semantic_gt_table.csv                           # 5,747 rows, semantic GT + provenance
├── semantic_metrics.csv
├── protocol_v2_vs_semantic.csv
├── gt_disagreement_summary.csv / .json
├── negative_pool_decisions_by_semantic.csv
├── iou_sensitivity/{iou_sensitivity_summary.csv, iou_sensitivity_per_detection.csv,
│                    semantic_rematch_by_threshold.csv, iou_sensitivity_info.json}
├── provenance/protected_outputs_baseline.json      # SHA-256 of protected outputs
├── EVALUATION_INFO.json                            # commit, manifest hash, status
└── README.md
```

Coverage: all 60 runs with Protocol v2 evaluations are re-evaluated (32 cover all 5,747
detections). Prediction-only runs without a Protocol v2 evaluation (smoke/qualification
runs and the partial `qwen2_5_vl/20260708_0020/A5`) are not included.
