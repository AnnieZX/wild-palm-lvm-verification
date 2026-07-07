# Evaluation Protocol

This document defines the official evaluation protocol for all experiments in this project.

## 1. Ground Truth

Ground truth annotations are LabelMe JSON files.

Every annotation with

```
label == "palm"
```

is converted into an axis-aligned bounding box by taking

```
xmin = min(x)
ymin = min(y)
xmax = max(x)
ymax = max(y)
```

This conversion is independent of LabelMe `shape_type` (rectangle, rotation, polygon, etc.).

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

This follows the Pascal VOC / COCO greedy matching convention.

## 3. Verification

Each matched or unmatched YOLO detection is passed to the vision-language model.

The model predicts one of:

- **Reliable**
- **Uncertain**
- **Unreliable**

For binary evaluation:

| Model prediction | Evaluation role |
|------------------|-----------------|
| Reliable | Positive prediction |
| Unreliable | Negative prediction |
| Uncertain | Excluded from binary evaluation (requires human verification) |

Binary verification metrics are computed only from definitive model decisions.

Predictions labeled "Uncertain" are intentionally excluded from Precision, Recall, F1-score, and Accuracy because they indicate that the vision-language model cannot make a reliable automatic decision.

These detections are considered candidates for manual human verification rather than automatic acceptance or rejection.

Ground-truth polarity is determined by greedy one-to-one IoU matching.

A matched detection (IoU >= 0.5) is considered a ground-truth positive.

An unmatched detection is considered a ground-truth negative.

Binary evaluation is then performed only for detections receiving a definitive model decision (Reliable or Unreliable).

Detections predicted as Uncertain are excluded from binary metrics and reported separately.

## 4. Metrics

### Detection

Report:

- TP
- FP
- FN
- Precision
- Recall
- F1
- Average IoU (of matched detection–GT pairs)
- Average YOLO confidence

### Verification

The following metrics are computed using only definitive predictions (Reliable and Unreliable):

- True Positive
- False Positive
- False Negative
- True Negative
- Precision
- Recall
- F1-score
- Accuracy

Also report separately:

- Number of Uncertain predictions
- Percentage of Uncertain predictions

The Uncertain rate reflects the proportion of detections requiring manual verification.

Additionally report the distribution of all model predictions:

- Reliable %
- Uncertain %
- Unreliable %

## 5. Experimental Consistency

The same evaluation protocol is applied consistently across all evaluated vision-language models.

Only definitive predictions (Reliable and Unreliable) participate in binary evaluation.

Predictions labeled Uncertain are excluded from binary metrics and instead represent cases requiring human review.

Only the verification model changes.

The dataset, matching algorithm, IoU threshold, and evaluation metrics remain identical across experiments.
