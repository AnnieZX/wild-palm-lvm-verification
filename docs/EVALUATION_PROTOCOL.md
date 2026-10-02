# Evaluation Protocol

This document defines the official evaluation protocol for all experiments in this project.

**Current version: Protocol v2** (effective 2026-09-27). Constant: `EVALUATION_PROTOCOL_VERSION = "v2"` in `src/preprocessing/gt_palm_bboxes.py`. Results: [EXPERIMENT_RESULTS_CANONICAL.md](EXPERIMENT_RESULTS_CANONICAL.md).

> [!IMPORTANT]
> **What Protocol v2 measures (terminology clarified 2026-10-01; algorithm unchanged).** Protocol v2 is an **annotation-alignment reference**: it asks whether a YOLO detection is matched to a LabelMe `palm` box (greedy one-to-one, IoU ≥ 0.5). It does **not** establish whether a detection is semantically a palm.
>
> Throughout this document and its outputs, the legacy notation is kept and means:
>
> | Legacy term | Meaning under Protocol v2 |
> |---|---|
> | GT+ / "positive" / `matched_gt = True` / `gt_label = positive` | **LabelMe-matched** detection (annotation-alignment positive) |
> | GT− / "negative" / `matched_gt = False` / `gt_label = negative` / `ground_truth_negative` | **LabelMe-unmatched** detection (annotation-alignment negative) — **not** a human-confirmed non-palm |
> | FP / `false_positive` | Reliable on a LabelMe-unmatched detection |
> | TN / `true_negative` | Unreliable on a LabelMe-unmatched detection |
> | Specificity | **Protocol-v2 alignment specificity** = share of decided LabelMe-unmatched detections classified Unreliable |
>
> Semantic validity is a separate construct. The official human review of all 638 LabelMe-unmatched detections found **619 palm, 19 ambiguous, 0 non-palm** (§8, [SEMANTIC_VALIDITY_AUDIT.md](SEMANTIC_VALIDITY_AUDIT.md)).

## 1. Ground Truth (LabelMe annotation-alignment reference)

The reference annotations ("ground truth" in legacy wording) are LabelMe JSON files (`/deac/csc/yangGrp/cuij/palm/Raw_Patches`, 880 files). They are used **for evaluation only** and are never shown to the model. They are treated as an alignment reference; they are not documented as exhaustive, so a real palm can be LabelMe-unmatched.

### 1.1 Palm-label normalization (v2)

A shape is a palm iff

```
isinstance(label, str) and label.strip().lower() == "palm"
```

implemented once as `is_palm_label()` in `src/preprocessing/gt_palm_bboxes.py` and used by every GT consumer. It accepts `palm`, `Palm`, `PALM`, and the same with surrounding whitespace; it rejects non-strings, empty labels and other words (`palms`, `palm tree`, …). Unit tests: `tests/test_gt_palm_bboxes.py`.

### 1.2 Box conversion

Every palm shape is converted into an axis-aligned bounding box by taking

```
xmin = min(x)
ymin = min(y)
xmax = max(x)
ymax = max(y)
```

This conversion is independent of LabelMe `shape_type` (rectangle, rotation, polygon, point, etc.). Point-shaped annotations produce zero-area boxes; they are **kept**, are counted as GT palms, and can never be matched (IoU = 0).

### 1.3 GT counts (Protocol v2)

| Quantity | Value |
|----------|------:|
| GT palm boxes (880 LabelMe files) | 5,853 (5,367 `palm` + 486 `Palm`; 3,306 rotation, 2,544 rectangle, 3 point) |
| Verification detections (YOLO confidence ≥ 0.5) | 5,747 |
| GT+ (LabelMe-matched) / GT− (LabelMe-unmatched) | **5,109 / 638** |
| Always-Reliable alignment accuracy (class prior) | 0.8890 |
| Human semantic audit of the 638 GT− | 619 palm / 19 ambiguous / 0 non-palm |

## 2. Detection Matching

Evaluation is performed independently for each image (patch).

All YOLO detections and all GT palm boxes are compared.

Pairwise IoU is computed between every detection and every GT.

Candidate pairs are sorted by descending IoU.

Greedy one-to-one matching is applied:

- Each detection may match at most one GT.
- Each GT may match at most one detection.

A match is accepted only if

```
IoU >= 0.5
```

This follows the Pascal VOC / COCO greedy matching convention. A detection with IoU ≥ 0.5 against a GT that is already taken by a better-overlapping detection is **unmatched** (GT−); 25 of the 5,747 verification detections are in this state (near-duplicate YOLO boxes).

Implementation: `src/evaluation/gt_matching.py`.

## 3. Verification

Each matched or unmatched YOLO detection is passed to the vision-language model.

The model predicts one of:

- **Reliable**
- **Uncertain**
- **Unreliable**

For binary evaluation:

| Model prediction | Evaluation role |
|------------------|-----------------|
| Reliable (R) | Positive prediction |
| Unreliable (Ur) | Negative prediction |
| Uncertain (U) | Excluded from binary evaluation (requires human verification) |

Alignment polarity is determined by greedy one-to-one IoU matching: a matched detection (IoU ≥ 0.5) is GT+ (LabelMe-matched), an unmatched detection is GT− (LabelMe-unmatched).

| | GT+ (LabelMe-matched) | GT− (LabelMe-unmatched) |
|---|---|---|
| **Reliable** | TP | FP |
| **Unreliable** | FN | TN |
| **Uncertain** | excluded | excluded |

Uncertain is never counted as FN or TN. These detections are candidates for manual human verification and are reported separately.

## 4. Metrics

### Detection

Report TP, FP, FN, Precision, Recall, F1, average IoU of matched pairs, and average YOLO confidence. Output: `outputs/evaluation_protocol_v2/detection_metrics.json` (`scripts/evaluate_detection_matching.py`).

### Verification (Protocol-v2 alignment metrics)

Computed using only definitive predictions (Reliable and Unreliable), against LabelMe alignment labels. Every metric in this table is an alignment metric; "Specificity" is reported as **Protocol-v2 alignment specificity** and must not be described as semantic non-palm specificity.

| Metric | Definition |
|--------|------------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Sensitivity (Recall) | TP / (TP + FN) |
| **Specificity** | **TN / (TN + FP)** |
| F1 | 2 · Precision · Sensitivity / (Precision + Sensitivity) |
| **Balanced Accuracy** | **(Sensitivity + Specificity) / 2** |

Specificity and balanced accuracy are written to every Protocol v2 `A*_metrics.json` by `scripts/compute_verification_metrics.py`.

Also report the full prediction distribution over all N detections: Reliable %, Uncertain %, Unreliable %, and the descriptors **coverage** = (R + Ur) / N and **abstention** = U / N. Because Accuracy and F1 exclude Uncertain and the class prior is high (0.889), they must always be read together with specificity, balanced accuracy and coverage.

## 5. Experimental Consistency

The same evaluation protocol is applied consistently across all evaluated vision-language models.

Only the verification model changes.

The dataset, matching algorithm, IoU threshold, palm-label rule and evaluation metrics remain identical across experiments.

## 6. Output Locations (versioned)

```
outputs/verification/<model_key>/<experiment_id>/<A1..A5>/sample_*.json            # predictions (protocol-independent)
outputs/evaluation_protocol_v2/<model_key>/<experiment_id>/<A1..A5>/<code>_evaluation.csv
outputs/evaluation_protocol_v2/<model_key>/<experiment_id>/<A1..A5>/<code>_metrics.json
outputs/evaluation_protocol_v2/<model_key>/<experiment_id>/PROTOCOL.json
outputs/evaluation_protocol_v2/{PROTOCOL.json, rescore_manifest.csv, detection_metrics.json}
outputs/evaluation/...                                                                # Protocol v1, frozen
```

- `src/paths.py`: `CURRENT_EVALUATION_ROOT = EVALUATION_PROTOCOL_V2_ROOT`; `EVALUATION_PROTOCOL_V1_ROOT = outputs/evaluation`.
- `scripts/evaluate_verification_against_groundtruth.py`, `scripts/compute_verification_metrics.py` and `jobs/run_verification.slurm` default to the v2 root; `jobs/run_verification.slurm` refuses to write into `outputs/evaluation/`.
- `scripts/rescore_protocol_v2.py` re-scores stored predictions into the v2 tree without inference and verifies that predictions, sample order and R/U/Ur are unchanged.
- `PROTOCOL.json` records the protocol version, GT rule, IoU threshold, matching rule, generation time, base git commit and whether the working tree was dirty.
- Legacy per-model Slurm scripts (header `DEPRECATED`) hardcode the v1 path `outputs/evaluation/`. Do not use them for new runs; use `scripts/submit_model_ablation.sh` → `jobs/run_verification.slurm`, or re-score stored predictions with `scripts/rescore_protocol_v2.py`.

Legacy pre-freeze Qwen2.5 paths under `outputs/verification/qwen/` remain valid for evaluation.

## 7. Protocol v1 (provenance)

Protocol v1 (July 2026 – 2026-09-27) was identical except for the label rule: `label == "palm"` (case-sensitive). It dropped the 486 `Palm` boxes in 70 files (parents 0194–0205), giving 5,367 GT boxes and GT+ 4,685 / GT− 1,062 (prior 0.8152). 424 detections flip GT− → GT+ under v2, none flip the other way. v1 outputs are frozen, unmodified, in `outputs/evaluation/`. Full correction record: [EXPERIMENT_RESULTS_CANONICAL.md §2 and §13](EXPERIMENT_RESULTS_CANONICAL.md#2-protocol-v1--v2-correction).

## 8. Semantic validity (separate construct)

Protocol v2 GT is annotation alignment: an unmatched detection is GT−, which does not by itself mean it is not a palm. Semantic validity is audited separately by human visual review and leaves this protocol, its labels and its outputs unchanged.

**Official result (2026-10-01):** all 638 Protocol-v2 LabelMe-unmatched detections were reviewed: **619 palm, 19 ambiguous, 0 non-palm**. A blind, confidence-stratified pilot of 400 lower-confidence YOLO detections (0.10–0.50, below this cohort) found 318 palm, 9 non-palm and 73 ambiguous. Canonical write-up, limitations and provenance: [SEMANTIC_VALIDITY_AUDIT.md](SEMANTIC_VALIDITY_AUDIT.md).

Consequences for this protocol:

- Protocol v2 metrics remain valid **as annotation-alignment metrics** and are not recomputed.
- Alignment specificity on this cohort cannot be read as non-palm rejection: semantic non-palm specificity is not estimable for the ≥ 0.5 cohort (0 confirmed non-palms).

The earlier semantic-GT evaluation design ([SEMANTIC_GT_EVALUATION.md](SEMANTIC_GT_EVALUATION.md), `scripts/evaluate_semantic_gt.py`) is **SUPERSEDED / NOT FOR CURRENT RESULTS**; it predates the official review workflow and does not consume its labels.

Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
