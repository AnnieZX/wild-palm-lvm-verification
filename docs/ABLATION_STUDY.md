# Ablation Study Design

This document describes the official ablation study design used throughout this project.

## Objective

The purpose of this ablation study is to investigate how different input information affects the ability of a vision-language model (VLM) to verify YOLO palm detections.

The detector (YOLO) is fixed throughout all experiments.

Only the input provided to the VLM changes.

The evaluation protocol remains identical across all experiments.

## Fixed Components

The following components remain unchanged for every experiment:

- Same YOLO detector
- Same detection results
- Same verification dataset
- Same LabelMe ground truth
- Same greedy one-to-one IoU matching
- Same IoU threshold (0.5)
- Same evaluation metrics
- Same verification prompt template except for the removed/added information

Only the VLM input is modified.

## Ablation Conditions

### A1 — Overlay Only

**Input:**

- Overlay image only

**Contains:**

- YOLO bounding box
- Palm ID

**Does NOT include:**

- Confidence score
- Geometry metadata
- Raw crop

**Purpose:**

Evaluate whether a simple visual overlay is sufficient for verification.

### A2 — Overlay + Confidence

**Input:**

- Overlay image
- YOLO confidence score

**Purpose:**

Evaluate whether detector confidence improves verification performance.

### A3 — Overlay + Confidence + Geometry

**Input:**

- Overlay image
- YOLO confidence
- Bounding-box geometry
- Area
- Width
- Height
- Aspect ratio
- Center coordinates

**Purpose:**

Evaluate whether explicit geometric information helps the VLM distinguish reliable detections. This is the richest metadata configuration.

### A4 — Dual Panel (Overlay + Crop) + Confidence

**Input:**

- Combined dual-panel image (full overlay + enlarged bbox crop)
- YOLO confidence in prompt metadata

**Purpose:**

Evaluate whether local crop detail helps when full overlay context is also available.

### A5 — Crop Only + Confidence

**Input:**

- Enlarged bbox crop only (no surrounding context)
- YOLO confidence in prompt metadata

No overlay visualization. No geometry metadata in the prompt.

**Purpose:**

Measure verification using crop appearance alone. Compare with A4 to test whether full-image context is necessary.

## Expected Findings

The hypothesis is:

A3 should achieve the best verification performance because it combines:

- visual context
- detector confidence
- geometric information

A5 is expected to perform worst because it removes all contextual information provided by the detector.

The study evaluates how much each additional information source contributes to verification performance.

## Experimental Fairness

Only one factor changes between ablation conditions.

All other components remain fixed.

Therefore, performance differences can be attributed to the availability of detector metadata rather than changes in the detector, dataset, matching algorithm, or evaluation protocol.

## Evaluation

All ablation experiments use the evaluation protocol defined in:

[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)

The same protocol is shared across all evaluated VLMs (`qwen2_5_vl`, LLaVA, Gemma 4, Qwen3-VL, etc.), ensuring fair comparison between ablation conditions and models.

See also: [ARCHITECTURE.md](ARCHITECTURE.md) · [FRAMEWORK_FREEZE.md](FRAMEWORK_FREEZE.md)
