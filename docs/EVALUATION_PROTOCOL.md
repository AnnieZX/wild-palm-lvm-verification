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

| Model prediction | Binary label |
|------------------|--------------|
| Reliable         | Positive     |
| Uncertain        | Negative     |
| Unreliable       | Negative     |

Ground-truth polarity for verification is determined by detection matching: a detection is **positive** if it was matched to a GT palm box (IoU ≥ 0.5); otherwise it is **negative**.

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

Report:

- TP
- FP
- FN
- Precision
- Recall
- F1
- Accuracy
- Reliable %
- Uncertain %
- Unreliable %

## 5. Experimental Consistency

The same evaluation protocol is used for all VLMs.

Only the verification model changes.

The dataset, matching algorithm, IoU threshold, and evaluation metrics remain identical across experiments.
