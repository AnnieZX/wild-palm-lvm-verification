# Detection-property behavior analysis (ground-truth independent)

> This analysis is ground-truth independent. It describes which stored detection properties are associated with VLM decision changes between input conditions. An association between a detection property and VLM behavior does **not** establish whether any decision, or any detection, is semantically correct.

**Research question.** Which detection-level properties are associated with VLM decision sensitivity to changes in visual input condition?

- **Primary comparison: A1→A5.** This is the clearest cross-model shift found in `docs/A1_A5_BEHAVIOR_ANALYSIS.md`.
- **Secondary comparison: A2→A5.** Both conditions carry the same confidence text, so this isolates the image change.

No inference was run, and no prediction, evaluation, matching, or semantic-review file was read for labels or modified. Specifically, the analysis uses no LabelMe annotation, no IoU, no GT+/GT−, no TP/FP/TN/FN, and no semantic palm/non-palm label.

| Item | Location |
|---|---|
| Computation | `scripts/analysis/detection_property_behavior_analysis.py` |
| Figures | `scripts/visualization/plot_detection_property_behavior.py` |
| Tests | `tests/test_detection_property_behavior_analysis.py` |
| Outputs | `outputs/detection_property_behavior_analysis/` (CSVs, `summary.md`, `analysis_info.json`, `figures/`) |

```bash
python scripts/analysis/detection_property_behavior_analysis.py     # ~20 s
python scripts/visualization/plot_detection_property_behavior.py
python -m unittest tests.test_detection_property_behavior_analysis -v
```

## 1. Audit of experiment construction

All 28,735 stored prompts (`outputs/verification_ablation_5747/<cond>/prompts/*.txt`) are byte-identical to the output of `src/prompts/ablation_verification_prompts.py`. The inference runner (`src/verification/jobs.py`) feeds exactly the `image_path` and `prompt_path` listed in each condition's `prompt_index.csv`. For 60 random detections, the overlay, A4, and A5 images were rebuilt from the raw patch pixels (no annotation files) with the repository functions, and all 60 were pixel-identical to the stored images.

Verified per-condition input (every source patch is 912×912 px):

| Cond. | Image fed to the model | Text metadata in prompt | Metadata instruction |
|---|---|---|---|
| A1 | 912×912 overlay: patch dimmed ×0.55 outside the box, box region at full brightness, green (0,200,0) 2-px rectangle | none | "Do not use YOLO confidence or bounding-box geometry." |
| A2 | same overlay as A1 | YOLO confidence (full float) | confidence auxiliary; no geometry |
| A3 | same overlay as A1 | confidence, box width, height, area (px), aspect ratio | confidence and geometry auxiliary |
| A4 | 1824×912 two-panel image: left = A1 overlay; right = overlay cropped to the box (no padding, includes the green border) and resized to 912×912 | YOLO confidence | confidence auxiliary |
| A5 | 512×512: **raw** patch (no dimming, **no green box**) cropped to the box with 15 px padding (clamped at patch edges), resized to 512×512 regardless of the crop's aspect ratio | YOLO confidence | confidence auxiliary |

Two construction facts matter here:

1. **The A5 prompt does not match the A5 image.** The shared "Important instructions" and decision definitions still say "The highlighted green bounding box indicates the candidate detection", but the A5 image contains no box (0 box-colored pixels).
2. **The crops are stretched to fixed sizes.** The A5 crop is always resized to 512×512, so non-square boxes are anisotropically distorted, and small boxes are enlarged more (median upscale 3.0×, range 1.3–7.0×). The A4 right panel is likewise stretched to 912×912.

## 2. Detection properties and provenance

All stored fields come from `outputs/verification_dataset/index.csv`. These fields agree exactly with the per-sample `metadata/*.json` for all 5,747 detections, and none is missing.

| Property | Source / field(s) | Stored or derived | Definition |
|---|---|---|---|
| YOLO confidence | `confidence` | stored | YOLO score (cohort filtered at ≥ 0.5) |
| Box coordinates | `bbox_x`, `bbox_y` | stored | top-left corner, px |
| Width / height | `bbox_width`, `bbox_height` | stored | px |
| Area | `bbox_area` | stored | = width × height (verified) |
| Center | `center_x`, `center_y` | stored | = x + w/2, y + h/2 (verified) |
| Source image ID | `image_name` | stored | raw patch stem |
| Source image size W, H | PNG header of `verification_dataset/images/<sample_id>.png` | derived from stored file | 912×912 for all; equals the raw-patch PNG header for all 870 patches |
| Normalized area | — | derived | `bbox_area / (W·H)` |
| Aspect ratio | — | derived | `w / h` |
| Elongation | — | derived | `|ln(w/h)|` (0 = square; symmetric for wide and tall boxes) |
| Normalized center | — | derived | `center_x / W`, `center_y / H` |
| Edge gap | — | derived | `min(x, y, W−x−w, H−y−h) / min(W,H)` (box-to-boundary distance) |
| Touches edge | — | derived | edge gap < 1 px |
| Center edge distance | — | derived | `min(cx, cy, W−cx, H−cy) / min(W,H)`, range (0, 0.5] |
| Detections in image | — | derived | number of cohort detections that share `image_name` |
| A5 crop size, upscale, distortion | — | derived with `src.preprocessing.ablation_verification_images._padded_bbox` | crop w, h (px); upscale = `sqrt(512² / (crop_w·crop_h))`; distortion = `|ln(crop_w / crop_h)|` |

All boxes lie inside the patch. 38.6% touch at least one patch edge.

**Excluded:** nothing proposed had to be excluded. Raw-patch pixel content was used only for the construction check in §1, not as a property.

**Collinearity** (Spearman ρ, `property_correlations.csv`):

- confidence vs. normalized area: 0.69;
- normalized area vs. A5 upscale: −0.99 (redundant);
- elongation vs. A5 aspect distortion: 0.96 (redundant);
- elongation vs. center edge distance: −0.49;
- center edge distance vs. edge gap: 0.91.

Boxes that touch an edge are more elongated (median elongation 0.43 vs. 0.15), consistent with truncation at the patch boundary. Because of this collinearity, the properties do not provide independent explanations.

## 3. Source-image grouping

| Level | Column | Clusters | Detections per cluster (min / median / mean / p90 / max) | Singletons |
|---|---|---|---|---|
| Source image | `image_name` | 870 | 1 / 6 / 6.6 / 12 / 20 | 69 |
| Parent frame (naming-derived) | `image_name` without the trailing `_<k>` | 268 | 1 / 11 / 21.4 / 53 / 102 | 10 |

Distribution of detections per image (number of images): 1:69, 2:87, 3:105, 4:60, 5:66, 6:77, 7:71, 8:60, 9:68, 10:44, 11:46, 12:38, 13:22, 14:23, 15:12, 16:9, 17:7, 18:1, 19:3, 20:2.

`image_name` identifies the source patch from which each detection, overlay, and crop was built, so it is the primary cluster. The patch names follow `<parent>_<k>` with k = 1–8. Patches sharing a parent may come from one larger frame and could overlap spatially, but this cannot be verified from stored data without annotation files. The parent prefix is therefore used only as a conservative sensitivity cluster. **Detections are not treated as independent.**

## 4. Outcomes and bootstrap

**Per-detection outcome.** For each model × detection × comparison (A1→A5, A2→A5), the analysis records:

- the exact transition: R→R, R→U, R→Ur, U→R, U→U, U→Ur, Ur→R, Ur→U, Ur→Ur;
- whether the decision changed;
- the direction, with the same definitions as the A1–A5 behavior analysis: *toward rejection* = R→U, R→Ur, U→Ur; *toward acceptance* = Ur→U, Ur→R, U→R.

The counts match `outputs/a1_a5_behavior_analysis/pairwise_transitions.csv` exactly; this is asserted at runtime. None of these categories is correct/incorrect or success/failure.

**Events and "at risk" sets.** Each rate is computed among the detections that could make the transition:

| Event | At risk | Counts as an event |
|---|---|---|
| changed, toward rejection, toward acceptance | all 5,747 | the stated change |
| `a->b` | detections with source decision *a* | target decision is *b* |
| `into_U` / `into_Ur` | detections not already U (or Ur) at the source condition | target decision is U (or Ur) |

**Bootstrap.**

- **Resampling:** each resample draws 870 source images with replacement, and every detection of a drawn image enters with that image's multiplicity. Individual detections are never resampled.
- **Settings:** 2,000 resamples, seed 20261001, percentile 95% intervals.
- **Sensitivity check:** a parent-frame bootstrap (268 clusters, seed 20261002) is reported alongside every contrast.
- **Bins:** cohort quintiles for continuous properties; fixed bins for confidence ([0.50, 0.60), [0.60, 0.70), [0.70, 0.80), [0.80, 0.85), [0.85, 0.90), [0.90, 1.00]) and for detections per image.
- **Contrasts:** highest minus lowest cohort quintile; for touches-edge, touching minus not touching.
- **Sparse flag:** contrasts with fewer than 50 detections at risk in either group are flagged `sparse_contrast`.

## 5. Results

Rates are percentages of the at-risk set. The brackets after each difference are the 95% image-clustered bootstrap interval (in percentage points, pp).

**Parent-frame sensitivity:** parent-frame intervals are at most 6.5 pp wider than image-level ones (`event_contrasts.csv`). Of 554 contrasts, 10 change whether their interval excludes 0, all with an endpoint within about 1 pp of 0. Examples: Qwen3-VL Ur→U by area; GLM-4.6V into Unreliable and R→Ur by elongation. These are labelled "borderline" below. No headline contrast is among them.

### 5.1 A1→A5: decision changed, lowest vs. highest quintile

| Model | Overall changed | Confidence Q1 → Q5 | Box area Q1 → Q5 | Elongation Q1 → Q5 | Touches edge: no → yes |
|---|---|---|---|---|---|
| Gemma-4 | 8.7 [8.0, 9.4] | 27.7 → 0.1 (−27.6 [−30.1, −25.0]) | 28.4 → 0.5 (−27.9 [−30.8, −25.1]) | 4.4 → 21.8 (+17.4) | 5.1 → 14.4 (+9.3) |
| GLM-4.6V | 48.0 [46.5, 49.4] | 60.6 → 24.1 (−36.5 [−40.4, −32.6]) | 60.1 → 26.3 (−33.8 [−38.0, −29.6]) | 42.0 → 57.9 (+15.9) | 44.9 → 52.9 (+8.0) |
| InternVL3.5 | 38.0 [36.6, 39.4] | 22.3 → 39.2 (+16.9 [13.2, 20.5]) | 20.0 → 43.0 (+23.0 [19.2, 26.9]) | 38.5 → 32.2 (−6.3) | 39.6 → 35.5 (−4.1) |
| Phi-4 | 23.0 [21.9, 24.1] | 47.2 → 3.3 (−43.9 [−47.2, −40.6]) | 48.5 → 3.8 (−44.7 [−48.0, −41.6]) | 16.5 → 42.6 (+26.1) | 17.0 → 32.6 (+15.6) |
| Qwen2.5-VL | 46.2 [44.6, 47.7] | 74.0 → 12.2 (−61.8 [−65.1, −58.6]) | 72.5 → 12.3 (−60.2 [−63.9, −56.5]) | 37.0 → 68.8 (+31.7) | 38.8 → 58.0 (+19.2) |
| Qwen3-VL | 73.4 [72.1, 74.7] | 75.4 → 48.8 (−26.6 [−30.8, −22.3]) | 78.6 → 50.4 (−28.2 [−32.2, −24.3]) | 67.9 → 80.0 (+12.1) | 70.0 → 78.9 (+8.9) |

The elongation and touches-edge intervals all exclude 0 (see `event_contrasts.csv`).

### 5.2 A2→A5 (same confidence text in both conditions): decision changed

| Model | Overall changed | Confidence Q1 → Q5 | Box area Q1 → Q5 |
|---|---|---|---|
| Gemma-4 | 8.6 [7.9, 9.4] | 26.8 → 0.1 (−26.7 [−29.2, −24.2]) | 29.0 → 0.5 (−28.5 [−31.5, −25.7]) |
| GLM-4.6V | 47.7 [46.2, 49.2] | 57.7 → 23.3 (−34.3 [−38.2, −30.4]) | 59.9 → 26.4 (−33.5 [−37.6, −29.4]) |
| InternVL3.5 | 40.3 [38.7, 41.7] | 15.8 → 40.0 (+24.2 [20.5, 27.9]) | 15.0 → 44.0 (+29.0 [25.5, 33.1]) |
| Phi-4 | 22.2 [21.1, 23.3] | 45.9 → 2.3 (−43.7 [−46.8, −40.6]) | 47.3 → 3.7 (−43.7 [−46.8, −40.4]) |
| Qwen2.5-VL | 46.3 [44.7, 47.9] | 83.0 → 9.7 (−73.4 [−76.2, −70.7]) | 82.5 → 9.9 (−72.6 [−75.6, −69.6]) |
| Qwen3-VL | 73.1 [71.7, 74.4] | 74.3 → 48.6 (−25.7 [−29.7, −21.5]) | 78.2 → 50.7 (−27.5 [−31.4, −23.6]) |

A2→A5 reproduces every A1→A5 direction, with similar magnitudes. Two models have steeper gradients in A2→A5:

- Qwen2.5-VL: −73 vs. −62 pp by confidence.
- InternVL3.5: +29 vs. +23 pp by area.

### 5.3 Model-specific transitions (A1→A5, area and confidence Q1 → Q5)

| Model | Event (at risk) | Overall | Area Q1 → Q5 | Confidence Q1 → Q5 |
|---|---|---|---|---|
| Qwen3-VL | R→U (A1 Reliable) | 67.4 [65.5, 69.1] | 82.6 → 43.1 (−39.6 [−44.5, −34.5]) | 80.0 → 43.0 (−37.0 [−41.6, −32.4]) |
| Qwen3-VL | Ur→U (A1 Unreliable) | 71.6 [68.9, 74.0] | 63.6 → 75.4 (+11.9 [0.3, 23.2], borderline) | 63.5 → 74.0 (+10.5 [−2.3, 22.1]) |
| Qwen3-VL | R→Ur (A1 Reliable) | 8.6 [7.6, 9.6] | 16.3 → 6.1 (−10.2 [−14.0, −6.9]) | 16.3 → 3.8 (−12.5 [−16.2, −8.9]) |
| InternVL3.5 | into U (A1 not Uncertain) | 56.1 [54.2, 57.9] | 89.5 → 41.9 (−47.6 [−52.6, −41.8]) | 82.4 → 36.4 (−46.0 [−51.5, −40.5]) |
| InternVL3.5 | R→U (A1 Reliable) | 54.8 [52.8, 56.7] | 83.2 → 41.9 (−41.3 [−48.3, −33.1]) | 76.9 → 36.4 (−40.5 [−47.1, −33.9]) |
| Qwen2.5-VL | into Ur (A1 not Unreliable) | 36.0 [34.4, 37.5] | 81.8 → 5.1 (−76.7 [−79.8, −73.6]) | 77.8 → 5.8 (−72.0 [−74.9, −69.0]) |
| Qwen2.5-VL | R→Ur (A1 Reliable) | 21.5 [19.9, 23.0] | 70.4 → 4.5 (−65.9 [−72.5, −58.6]) | 64.7 → 4.4 (−60.3 [−65.5, −55.1]) |
| Qwen2.5-VL | U→Ur (A1 Uncertain) | 65.4 [62.9, 67.9] | 85.5 → 13.3 (−72.1 [−80.1, −63.4]) | 85.0 → 29.7 (−55.3 [−66.7, −43.8]) |
| GLM-4.6V | into U | 18.7 [17.5, 19.8] | 41.4 → 4.4 (−37.0 [−40.2, −33.8]) | 38.9 → 2.4 (−36.4 [−39.7, −33.5]) |
| GLM-4.6V | into Ur | 36.3 [34.7, 37.9] | 42.2 → 20.2 (−22.1 [−27.4, −16.6]) | 47.8 → 16.7 (−31.0 [−36.2, −25.9]) |
| Gemma-4 | R→Ur (A1 Reliable) | 7.4 [6.7, 8.1] | 26.5 → 0.5 (−26.0 [−29.1, −23.2]) | 25.5 → 0.1 (−25.4 [−28.1, −22.7]) |
| Gemma-4 | Ur→R (A1 Unreliable, n = 164) | 54.3 [46.2, 61.3] | no A1-Unreliable detections in Q5 (sparse) | no A1-Unreliable detections in Q5 (sparse) |
| Phi-4 | R→Ur (A1 Reliable) | 21.5 [20.3, 22.7] | 61.4 → 2.2 (−59.2 [−62.7, −55.8]) | 56.6 → 1.1 (−55.5 [−58.9, −52.1]) |
| Phi-4 | Ur→R (A1 Unreliable) | 32.7 [29.5, 36.2] | 21.0 → 65.5 (+44.5 [25.1, 62.3]; Q5 n = 29, sparse) | 18.2 → 70.3 (+52.1 [34.7, 67.9]; Q5 n = 37, sparse) |

**Edge and elongation, A1→A5** (Q1 → Q5 of center edge distance, i.e. near the edge → far from it; then elongation Q1 → Q5):

| Model | Event | Center edge distance | Elongation |
|---|---|---|---|
| GLM-4.6V | into U | 39.1 → 11.6 | 12.5 → 35.3 |
| GLM-4.6V | into Ur | 39.8 → 33.4 (−6.4 [−12.6, −0.7]) | 32.4 → 38.3 (+5.8 [0.5, 11.2], borderline) |
| InternVL3.5 | into U | 83.4 → 50.9 | 48.6 → 75.7 |
| Qwen3-VL | R→U | 82.9 → 62.9 | 61.2 → 80.1 |
| Qwen2.5-VL | into Ur | 77.7 → 23.3 | 26.3 → 66.8 |
| Phi-4 | R→Ur | 56.7 → 12.7 | 13.4 → 48.3 |
| Gemma-4 | R→Ur | 24.8 → 2.7 | 3.1 → 21.1 |

GLM-4.6V's move into Uncertain is concentrated near edges and on elongated boxes; its move into Unreliable is only weakly related to either.

**GLM-4.6V, A5 endpoint profile** (all detections ending in each A5 label; medians):

| A5 label | n | Median confidence | Median area | Median elongation | Median center edge distance |
|---|---|---|---|---|---|
| Uncertain | 1082 | 0.720 | 1.54% | 0.35 | 0.068 |
| Unreliable | 2502 | 0.805 | 2.14% | 0.23 | 0.107 |
| Reliable | 2163 | 0.877 | 3.53% | 0.16 | 0.166 |

### 5.4 Median property by direction (A1→A5)

Unchanged vs. shifted detections, with the image-clustered interval for the difference (`median_differences.csv`):

| Model | Unchanged: confidence / area | Toward rejection: confidence / area | Toward acceptance: confidence / area |
|---|---|---|---|
| Gemma-4 | 0.844 / 2.59% | 0.637 / 1.14% | 0.665 / 1.25% |
| GLM-4.6V | 0.858 / 2.88% | 0.814 / 2.24% | 0.771 / 1.75% |
| InternVL3.5 | 0.817 / 2.18% | 0.855 / 2.90% | 0.815 / 1.80% |
| Phi-4 | 0.853 / 2.80% | 0.702 / 1.51% | 0.825 / 1.76% |
| Qwen2.5-VL | 0.867 / 3.22% | 0.742 / 1.69% | 0.834 / 2.29% |
| Qwen3-VL | 0.872 / 3.41% | 0.835 / 2.42% | 0.737 / 1.66% |

Every confidence and area difference from "unchanged" has an interval excluding 0, except InternVL3.5's toward-acceptance confidence difference (−0.001 [−0.031, 0.020]).

### 5.5 Separating correlated properties (A1→A5 changed rate, `stratified_change_rates.csv`)

- **Area within confidence tertiles:** the decreasing gradient across area quintiles persists within confidence strata for Gemma-4, Phi-4, Qwen2.5-VL, and GLM-4.6V (middle and high tertiles), and for Qwen3-VL (high tertile).
- **Confidence within area tertiles:** the confidence gradient persists within area strata for the same models. For example, Qwen2.5-VL in the middle area tertile goes from 0.80 (confidence Q1) to 0.24 (Q5); Phi-4 in the high area tertile goes from 0.21 to 0.02.
- **Some strata are thin:** the low-confidence / large-box and high-confidence / small-box cells are small (as few as 4 detections).
- **Elongation within edge strata:** elongation's association is much weaker once edge-touching is held fixed, and is concentrated in the most elongated quintile. For Qwen2.5-VL, elongation Q1 → Q5 goes 0.35 → 0.64 for boxes not touching an edge and 0.45 → 0.70 for boxes touching an edge.
- **Position:** horizontal and vertical position show only a U-shape at the patch edges (Q1 and Q5 higher). This is already captured by edge distance, so no separate position figure was made.
- **Detections per image:** no consistent association. InternVL3.5 is the exception, with higher A1→A5 change in crowded images: +13.9 pp [8.3, 19.1] for 13–20 vs. 1–3 detections per image.

## 6. Figures (`outputs/detection_property_behavior_analysis/figures/`)

- `change_rate_by_confidence.png`: A1→A5 and A2→A5 changed rate by confidence bin, all models.
- `change_rate_by_box_area.png`: the same by normalized-area quintile.
- `into_uncertain_qwen3_internvl.png`: Qwen3-VL and InternVL3.5 transitions into Uncertain vs. area and confidence.
- `into_unreliable_qwen2_5.png`: Qwen2.5-VL transitions into Unreliable vs. area and confidence.
- `glm_gemma_phi_transitions.png`: GLM-4.6V into Uncertain vs. into Unreliable; Gemma-4 and Phi-4 Reliable → Unreliable.

All bands are 95% image-clustered bootstrap intervals.

## 7. Observations

What the stored data directly show:

1. **Small boxes and low YOLO confidence go with larger A1→A5 movement.** Removing context is associated with much larger decision movement for small, low-confidence detections in five of six models: Gemma-4, GLM-4.6V, Phi-4, Qwen2.5-VL, Qwen3-VL. The lowest-vs-highest quintile differences in change rate range from about −27 to −62 pp. The same pattern appears in A2→A5, so it does not depend on the confidence text being added.
2. **The direction of movement on small, low-confidence boxes depends on the model.** It is into Uncertain for Qwen3-VL (R→U 83% in the smallest quintile vs. 43% in the largest) and InternVL3.5 (into U 90% vs. 42%). It is into Unreliable for Qwen2.5-VL (82% vs. 5%), Phi-4 (R→Ur 61% vs. 2%), and Gemma-4 (R→Ur 27% vs. 0.5%). GLM-4.6V moves into both, but its abstention is more concentrated on small, edge, and elongated boxes than its rejection.
3. **Large, high-confidence boxes are comparatively stable** for most models: Gemma-4 changes 0.5%, Phi-4 3.8%, and Qwen2.5-VL 12.3% of decisions in the largest-area quintile. Qwen3-VL is the exception: it still moves 43% of A1-Reliable decisions in the largest quintile to Uncertain.
4. **InternVL3.5's raw change rate rises with box size**, but this reflects its A1 baseline. Most small boxes are already Uncertain at A1 and so cannot change into Uncertain. Among detections not already Uncertain, the rate of moving into Uncertain falls with box size (90% → 42%), the same direction as the other models.
5. **Boxes touching the patch edge and elongated boxes change more often** for Gemma-4, GLM-4.6V, Phi-4, Qwen2.5-VL, and Qwen3-VL. These two properties largely describe the same boxes (edge-truncated boxes are elongated), and part of their association remains after stratifying by edge-touching.
6. **A few events run against the dominant gradient.** Qwen3-VL's Ur→U and Phi-4's Ur→R are more frequent for larger, higher-confidence boxes. Some of these contrasts are sparse.
7. **Image clustering changes little.** Image-clustered and parent-frame-clustered intervals lead to the same conclusions for all headline associations. Only 10 of 554 contrasts, all borderline, change whether their interval excludes 0.

## 8. Interpretation

Plausible, untested behavioral readings:

- **Confidence is unlikely to act through the text.** A2→A5 holds the confidence text fixed, yet confidence remains associated with A5 change. Confidence is also correlated with box size and edge position, so its association is more plausibly explained by what low-confidence detections look like (small, truncated, ambiguous crops) than by the model reading the number.
- **The A5 crop changes the evidence most for small and truncated boxes.** These are upscaled the most (up to 7×), lose the surrounding context that may have supported the A1 decision, and, if elongated, are stretched the most. Upscaling, aspect distortion, small object size, and low detector confidence cannot be separated with these data (ρ up to 0.99).
- **Models differ in what they do with weaker visual evidence.** Some abstain (Qwen3-VL, InternVL3.5), some reject (Qwen2.5-VL, Phi-4, Gemma-4), and GLM-4.6V does both. This sharpens the cross-model contrast reported in `docs/A1_A5_BEHAVIOR_ANALYSIS.md`: the A5 shift is concentrated on the same kind of detection in every model, but is expressed through different output labels.
- **The A5 prompt still refers to a green box that is absent.** This mismatch may contribute to the A5 shift for every detection. It is a design confound, not something this analysis can measure.

## 9. Cannot conclude

- **No property implies correctness or incorrectness.** Association between detection properties and VLM behavior does not establish semantic correctness. Nothing here shows that small, low-confidence, edge-touching, or elongated detections are false detections, or that large, high-confidence detections are true ones.
- **No condition or behavior is shown to be appropriate.** It cannot be said whether A5 abstention or rejection on small boxes is appropriate, or whether the A1 decisions on those boxes were.
- **No causal effect of a single property can be claimed.** Confidence, box size, A5 upscaling, edge truncation, and elongation are correlated, and none was manipulated.
- **Run-to-run variability is unknown.** Each model × condition is a single stored run, so the intervals reflect resampling of source images, not decoding variability.
- **No significance testing.** The intervals describe precision under image-level resampling; they are not hypothesis tests, and many contrasts were examined.

## 10. Data-integrity notes

- **Qwen2.5-VL input timing.** The stored ablation inputs and `verification_dataset/index.csv` were last written on 2026-07-11, after the Qwen2.5-VL A1–A4 run (2026-07-08). Several facts indicate the inputs it used were the same:
  - the only code changes between the two dates in dataset or prompt generation were docstrings;
  - `predictions_full.json` dates from 2026-06-28;
  - generation is deterministic;
  - all 11,990 Qwen2.5-VL A2–A5 responses that quote a confidence value match the stored confidence for that `sample_id`.

  The July 8 inputs themselves cannot be byte-verified.
- **Parent-frame grouping** is inferred from file naming only.
- **No missing, duplicate, or out-of-range property values** were found. All transition counts match the A1–A5 behavior analysis.

## 11. Validation

Runtime assertions and `tests/test_detection_property_behavior_analysis.py` (17 tests) cover:

- 5,747-row cohort, unique IDs, and identical cohort across inputs;
- complete property join;
- property ranges, including normalized coordinates in (0, 1) and edge distances ≥ 0;
- hand-computed derived values;
- transition counts equal to the A1–A5 behavior analysis, with a tampering check that must fail;
- event at-risk and event definitions;
- bins partitioning each at-risk set;
- the bootstrap resamples whole images: weighted statistics equal an explicit whole-image resample, multiplicities sum to the number of images, and the seed is deterministic;
- no ground-truth-like column in any input or output;
- no ground-truth or LabelMe module or path referenced by the script;
- the output-directory guard;
- input SHA-256 digests unchanged after the run.

The protected-output hash snapshot (`SEMANTIC_GT_VERIFY_SNAPSHOT=1`) was re-run after this analysis.
