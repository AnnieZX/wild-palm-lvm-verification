# A1–A5 VLM behavior analysis (ground-truth independent)

> This analysis is ground-truth independent. It characterizes how VLM decisions change as the visual input condition changes; it does not measure whether those decisions are semantically correct.

**Research question.** How do the A1–A5 input conditions affect VLM verification behavior on the fixed cohort of 5,747 YOLO detections?

The analysis uses only the stored `decision` field (Reliable / Uncertain / Unreliable) of existing prediction JSONs. It does not use LabelMe annotations, IoU matching, GT+/GT−, the semantic review, or any correctness metric. No inference was run and no stored prediction, evaluation, or review file was modified.

| Item | Location |
|---|---|
| Computation | `scripts/analysis/a1_a5_behavior_analysis.py` |
| Figures | `scripts/visualization/plot_a1_a5_behavior.py` |
| Tests | `tests/test_a1_a5_behavior_analysis.py` |
| Outputs | `outputs/a1_a5_behavior_analysis/` (CSV tables, `tables.md`, `analysis_info.json`, `figures/`) |

```bash
python scripts/analysis/a1_a5_behavior_analysis.py      # ~1 min, reads 172,410 JSONs
python scripts/visualization/plot_a1_a5_behavior.py
python -m unittest tests.test_a1_a5_behavior_analysis -v
```

## 1. Input conditions

All five conditions use the same detections and the same base prompt; they differ in image and metadata.

| Code | Image | Text metadata |
|---|---|---|
| A1 | full patch, dimmed background, green box | none (prompt says not to use confidence or geometry) |
| A2 | same as A1 | YOLO confidence |
| A3 | same as A1 | YOLO confidence + box width/height/area/aspect ratio |
| A4 | two panels: A1 image + enlarged crop | YOLO confidence |
| A5 | enlarged crop only, no surrounding context | YOLO confidence |

A1 is the canonical reference, but A1→A4 and A1→A5 change both the image and the text (confidence is added). A2→A4 and A2→A5 keep the text fixed and change only the image; they are reported in the pairwise extension.

## 2. Input audit

All 67 `<model>/<experiment>/<A*>` prediction directories under `outputs/verification/` were inventoried. A run was eligible only if all five conditions:

- contain exactly 5,747 prediction JSONs;
- have unique `sample_id`s, each equal to its filename;
- have the same `sample_id` set as `outputs/verification_dataset/index.csv`;
- have a `results_index.csv` with the same IDs and all rows `status == ok`;
- have no `parse_error` or `inference_error`;
- have every `decision` in {Reliable, Uncertain, Unreliable}.

| Model | Prediction directory (`outputs/verification/…`) | Eligible |
|---|---|---|
| Gemma-4-12B-it | `gemma4/20260925_gemma4_A1A5_5747/A1…A5` | yes |
| GLM-4.6V-Flash | `glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747/A1…A5` | yes |
| InternVL3.5-8B-HF | `internvl3_5_hf/20260924_internvl3_5_hf_A1A5_5747/A1…A5` | yes |
| Phi-4-multimodal | `phi4_multimodal/20260921_phi4_A1A5_5747/A1…A5` | yes |
| Qwen2.5-VL-7B | `qwen/20260708_0020/A1…A5` (legacy tree) | yes |
| Qwen3-VL-8B | `qwen3_vl/qwen3vl_A1A5_5747/A1…A5` | yes |
| MiniCPM-V-4.5 | `minicpm_v4_5/20260923_minicpm_A1A5_5747` | no: A1 only at full scale |
| Molmo2-8B | `molmo2_8b/20260923_molmo2_A1A5_5747` | no: A1 only at full scale |
| — | `qwen2_5_vl/20260708_0020/A5` | no: cancelled wrong-path job 8318954 (853 files, no index) |

All other runs are 1–1,000-sample qualification or subset runs.

**Provenance notes.**

- **Qwen2.5-VL A5 was resumed.** `sample_000001…002585` come from the original job and `sample_002586…005747` from resume job 8318977; the split is contiguous and has no overlap.
- **Legacy Qwen2.5-VL JSONs have no `model_key`, `condition`, or `experiment_id` fields** (except the resumed A5 files), so for this model the condition is identified by directory only.
- **Cross-check against Protocol v2.** For all 30 model-conditions, Table B counts match the prediction column (`verification_label`) of the stored Protocol v2 evaluation CSVs. Only that prediction column was read.
- **Fingerprints.** Per-condition SHA-256 fingerprints of the sorted `(sample_id, decision)` pairs are recorded in `analysis_info.json`.

**Field used.** The stored `decision` field is used as-is; no raw response is re-parsed.

## 3. Definitions

- **Decided** = Reliable + Unreliable. **Abstention rate** = Uncertain / N.
- **Transition matrix**: a per-detection 3×3 cross-tabulation of the decision under condition *a* (row) against condition *b* (column), on identical `sample_id`s.
- **Direction** uses the order Reliable → Uncertain → Unreliable. *Toward rejection* = R→U, R→X, U→X. *Toward acceptance* = X→U, X→R, U→R (X = Unreliable). These are names for the direction of a behavioral shift; neither direction is better or worse. Unchanged + toward rejection + toward acceptance = N.
- **Net toward rejection** = toward rejection − toward acceptance, as a share of N.
- **Changed into L** = number of changed detections whose new label is L.
- **Stability**: per detection, the number of distinct labels over A1–A5 (1, 2, or 3), and the number of adjacent changes along A1→A2→A3→A4→A5 (0–4). Adjacent changes depend on this ordering, which is not a single experimental dimension; distinct-label counts do not depend on it. The **label variability rate** is the share of detections with more than one distinct label. It is a descriptive count, not a new metric.

## 4. Results

All percentages are of N = 5,747. The full tables, including all ten pairwise 3×3 matrices, are in `outputs/a1_a5_behavior_analysis/tables.md` and the CSVs.

### Table B — Decision distributions

| Model | Cond. | Reliable | Uncertain | Unreliable | Reliable % | Uncertain % | Unreliable % | Decided | Abstention |
|---|---|---|---|---|---|---|---|---|---|
| Gemma-4-12B-it | A1 | 5583 | 0 | 164 | 97.15 | 0.00 | 2.85 | 5747 | 0.0000 |
| Gemma-4-12B-it | A2 | 5595 | 0 | 152 | 97.36 | 0.00 | 2.64 | 5747 | 0.0000 |
| Gemma-4-12B-it | A3 | 5424 | 0 | 323 | 94.38 | 0.00 | 5.62 | 5747 | 0.0000 |
| Gemma-4-12B-it | A4 | 5547 | 0 | 200 | 96.52 | 0.00 | 3.48 | 5747 | 0.0000 |
| Gemma-4-12B-it | A5 | 5261 | 0 | 486 | 91.54 | 0.00 | 8.46 | 5747 | 0.0000 |
| GLM-4.6V-Flash | A1 | 3719 | 50 | 1978 | 64.71 | 0.87 | 34.42 | 5697 | 0.0087 |
| GLM-4.6V-Flash | A2 | 3783 | 57 | 1907 | 65.83 | 0.99 | 33.18 | 5690 | 0.0099 |
| GLM-4.6V-Flash | A3 | 3696 | 100 | 1951 | 64.31 | 1.74 | 33.95 | 5647 | 0.0174 |
| GLM-4.6V-Flash | A4 | 3960 | 64 | 1723 | 68.91 | 1.11 | 29.98 | 5683 | 0.0111 |
| GLM-4.6V-Flash | A5 | 2163 | 1082 | 2502 | 37.64 | 18.83 | 43.54 | 4665 | 0.1883 |
| InternVL3.5-8B-HF | A1 | 3085 | 2570 | 92 | 53.68 | 44.72 | 1.60 | 3177 | 0.4472 |
| InternVL3.5-8B-HF | A2 | 3497 | 2239 | 11 | 60.85 | 38.96 | 0.19 | 3508 | 0.3896 |
| InternVL3.5-8B-HF | A3 | 3420 | 2298 | 29 | 59.51 | 39.99 | 0.50 | 3449 | 0.3999 |
| InternVL3.5-8B-HF | A4 | 3343 | 2399 | 5 | 58.17 | 41.74 | 0.09 | 3348 | 0.4174 |
| InternVL3.5-8B-HF | A5 | 1515 | 4078 | 154 | 26.36 | 70.96 | 2.68 | 1669 | 0.7096 |
| Phi-4-multimodal | A1 | 4985 | 0 | 762 | 86.74 | 0.00 | 13.26 | 5747 | 0.0000 |
| Phi-4-multimodal | A2 | 5012 | 0 | 735 | 87.21 | 0.00 | 12.79 | 5747 | 0.0000 |
| Phi-4-multimodal | A3 | 4132 | 0 | 1615 | 71.90 | 0.00 | 28.10 | 5747 | 0.0000 |
| Phi-4-multimodal | A4 | 3481 | 0 | 2266 | 60.57 | 0.00 | 39.43 | 5747 | 0.0000 |
| Phi-4-multimodal | A5 | 4160 | 0 | 1587 | 72.39 | 0.00 | 27.61 | 5747 | 0.0000 |
| Qwen2.5-VL-7B | A1 | 3593 | 1775 | 379 | 62.52 | 30.89 | 6.59 | 3972 | 0.3089 |
| Qwen2.5-VL-7B | A2 | 4268 | 1344 | 135 | 74.26 | 23.39 | 2.35 | 4403 | 0.2339 |
| Qwen2.5-VL-7B | A3 | 4416 | 1313 | 18 | 76.84 | 22.85 | 0.31 | 4434 | 0.2285 |
| Qwen2.5-VL-7B | A4 | 3570 | 1557 | 620 | 62.12 | 27.09 | 10.79 | 4190 | 0.2709 |
| Qwen2.5-VL-7B | A5 | 3065 | 455 | 2227 | 53.33 | 7.92 | 38.75 | 5292 | 0.0792 |
| Qwen3-VL-8B | A1 | 4309 | 246 | 1192 | 74.98 | 4.28 | 20.74 | 5501 | 0.0428 |
| Qwen3-VL-8B | A2 | 4636 | 401 | 710 | 80.67 | 6.98 | 12.35 | 5346 | 0.0698 |
| Qwen3-VL-8B | A3 | 4221 | 367 | 1159 | 73.45 | 6.39 | 20.17 | 5380 | 0.0639 |
| Qwen3-VL-8B | A4 | 5051 | 276 | 420 | 87.89 | 4.80 | 7.31 | 5471 | 0.0480 |
| Qwen3-VL-8B | A5 | 1081 | 3948 | 718 | 18.81 | 68.70 | 12.49 | 1799 | 0.6870 |

### Table C — Paired transitions, A1 reference

| Model | Comparison | Unchanged | Changed | Toward rejection | Toward acceptance | Net toward rejection (pp) |
|---|---|---|---|---|---|---|
| Gemma-4-12B-it | A1→A2 | 5675 (98.75%) | 72 (1.25%) | 30 (0.52%) | 42 (0.73%) | −0.21 |
| Gemma-4-12B-it | A1→A3 | 5570 (96.92%) | 177 (3.08%) | 168 (2.92%) | 9 (0.16%) | +2.77 |
| Gemma-4-12B-it | A1→A4 | 5477 (95.30%) | 270 (4.70%) | 153 (2.66%) | 117 (2.04%) | +0.63 |
| Gemma-4-12B-it | A1→A5 | 5247 (91.30%) | 500 (8.70%) | 411 (7.15%) | 89 (1.55%) | +5.60 |
| GLM-4.6V-Flash | A1→A2 | 5458 (94.97%) | 289 (5.03%) | 107 (1.86%) | 182 (3.17%) | −1.30 |
| GLM-4.6V-Flash | A1→A3 | 5359 (93.25%) | 388 (6.75%) | 192 (3.34%) | 196 (3.41%) | −0.07 |
| GLM-4.6V-Flash | A1→A4 | 4423 (76.96%) | 1324 (23.04%) | 540 (9.40%) | 784 (13.64%) | −4.25 |
| GLM-4.6V-Flash | A1→A5 | 2990 (52.03%) | 2757 (47.97%) | 1904 (33.13%) | 853 (14.84%) | +18.29 |
| InternVL3.5-8B-HF | A1→A2 | 5126 (89.19%) | 621 (10.81%) | 64 (1.11%) | 557 (9.69%) | −8.58 |
| InternVL3.5-8B-HF | A1→A3 | 5145 (89.53%) | 602 (10.47%) | 102 (1.77%) | 500 (8.70%) | −6.93 |
| InternVL3.5-8B-HF | A1→A4 | 4718 (82.09%) | 1029 (17.91%) | 343 (5.97%) | 686 (11.94%) | −5.97 |
| InternVL3.5-8B-HF | A1→A5 | 3561 (61.96%) | 2186 (38.04%) | 1846 (32.12%) | 340 (5.92%) | +26.20 |
| Phi-4-multimodal | A1→A2 | 5530 (96.22%) | 217 (3.78%) | 95 (1.65%) | 122 (2.12%) | −0.47 |
| Phi-4-multimodal | A1→A3 | 4886 (85.02%) | 861 (14.98%) | 857 (14.91%) | 4 (0.07%) | +14.84 |
| Phi-4-multimodal | A1→A4 | 4083 (71.05%) | 1664 (28.95%) | 1584 (27.56%) | 80 (1.39%) | +26.17 |
| Phi-4-multimodal | A1→A5 | 4424 (76.98%) | 1323 (23.02%) | 1074 (18.69%) | 249 (4.33%) | +14.36 |
| Qwen2.5-VL-7B | A1→A2 | 4828 (84.01%) | 919 (15.99%) | 36 (0.63%) | 883 (15.36%) | −14.74 |
| Qwen2.5-VL-7B | A1→A3 | 4596 (79.97%) | 1151 (20.03%) | 23 (0.40%) | 1128 (19.63%) | −19.23 |
| Qwen2.5-VL-7B | A1→A4 | 4106 (71.45%) | 1641 (28.55%) | 926 (16.11%) | 715 (12.44%) | +3.67 |
| Qwen2.5-VL-7B | A1→A5 | 3093 (53.82%) | 2654 (46.18%) | 2158 (37.55%) | 496 (8.63%) | +28.92 |
| Qwen3-VL-8B | A1→A2 | 5170 (89.96%) | 577 (10.04%) | 25 (0.43%) | 552 (9.61%) | −9.17 |
| Qwen3-VL-8B | A1→A3 | 5210 (90.66%) | 537 (9.34%) | 292 (5.08%) | 245 (4.26%) | +0.82 |
| Qwen3-VL-8B | A1→A4 | 4297 (74.77%) | 1450 (25.23%) | 309 (5.38%) | 1141 (19.85%) | −14.48 |
| Qwen3-VL-8B | A1→A5 | 1527 (26.57%) | 4220 (73.43%) | 3323 (57.82%) | 897 (15.61%) | +42.21 |

The 3×3 counts for every row above are in `table_c_transitions_a1_reference.csv` and `tables.md`.

### Pairwise extension — changed % for every condition pair

| Model | A1→A2 | A1→A3 | A1→A4 | A1→A5 | A2→A3 | A2→A4 | A2→A5 | A3→A4 | A3→A5 | A4→A5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-4-12B-it | 1.25 | 3.08 | 4.70 | 8.70 | 3.15 | 4.59 | 8.63 | 5.38 | 8.27 | 6.23 |
| GLM-4.6V-Flash | 5.03 | 6.75 | 23.04 | 47.97 | 4.35 | 21.39 | 47.75 | 20.92 | 47.33 | 47.54 |
| InternVL3.5-8B-HF | 10.81 | 10.47 | 17.91 | 38.04 | 4.65 | 14.06 | 40.26 | 14.04 | 39.38 | 38.04 |
| Phi-4-multimodal | 3.78 | 14.98 | 28.95 | 23.02 | 15.35 | 28.62 | 22.20 | 23.06 | 23.14 | 17.66 |
| Qwen2.5-VL-7B | 15.99 | 20.03 | 28.55 | 46.18 | 6.61 | 26.73 | 46.32 | 27.49 | 47.52 | 39.88 |
| Qwen3-VL-8B | 10.04 | 9.34 | 25.23 | 73.43 | 10.96 | 20.71 | 73.08 | 25.84 | 71.78 | 76.23 |

Net toward rejection for the image-only comparisons (text held fixed): A2→A4 is Gemma +0.84, GLM −3.05, InternVL +2.58, Phi-4 +26.64, Qwen2.5-VL +17.92, Qwen3-VL −7.73 pp. A2→A5 is Gemma +5.81, GLM +19.66, InternVL +34.66, Phi-4 +14.83, Qwen2.5-VL +37.97, Qwen3-VL +54.92 pp.

### Per-detection stability across A1–A5

| Model | Identical A1–A5 | Exactly 2 labels | All 3 labels | Adjacent changes (total) | Mean / detection | Always R | Always U | Always X |
|---|---|---|---|---|---|---|---|---|
| Gemma-4-12B-it | 5106 (88.85%) | 641 (11.15%) | 0 (0.00%) | 920 | 0.160 | 5076 | 0 | 30 |
| GLM-4.6V-Flash | 2460 (42.80%) | 2884 (50.18%) | 403 (7.01%) | 4473 | 0.778 | 1746 | 3 | 711 |
| InternVL3.5-8B-HF | 2878 (50.08%) | 2832 (49.28%) | 37 (0.64%) | 3881 | 0.675 | 1203 | 1675 | 0 |
| Phi-4-multimodal | 3468 (60.34%) | 2279 (39.66%) | 0 (0.00%) | 3439 | 0.598 | 3027 | 0 | 441 |
| Qwen2.5-VL-7B | 2514 (43.74%) | 2350 (40.89%) | 883 (15.36%) | 5171 | 0.900 | 2439 | 62 | 13 |
| Qwen3-VL-8B | 1092 (19.00%) | 3624 (63.06%) | 1031 (17.94%) | 7073 | 1.231 | 1010 | 22 | 60 |

### Figures (`outputs/a1_a5_behavior_analysis/figures/`)

- `decision_distribution_by_condition.png`: Reliable / Uncertain / Unreliable shares per condition, one panel per model.
- `transition_heatmaps_A1_to_A5.png`: paired A1→A5 3×3 matrices (counts, shaded by row share).
- `decision_change_from_A1.png`: changed %, toward-rejection %, and toward-acceptance % for A1→A2…A5.

## 5. Observations (descriptive only)

1. **Overall sensitivity to condition differs by model.** Qwen3-VL changes most: only 19.0% of detections keep the same label across A1–A5, and 73.4% change from A1 to A5. Gemma-4 changes least: 88.9% identical and 8.7% changed from A1 to A5. Ordered by label variability rate: Qwen3-VL 0.81, GLM 0.57, Qwen2.5-VL 0.56, InternVL 0.50, Phi-4 0.40, Gemma-4 0.11.
2. **Two models never output Uncertain.** Gemma-4 and Phi-4 gave 0 Uncertain decisions in every condition, so all of their changes are Reliable↔Unreliable.
3. **Rising Uncertain.** From A1 to A5, Uncertain rises for Qwen3-VL (4.3% → 68.7%), InternVL (44.7% → 71.0%), and GLM (0.9% → 18.8%). For all three it stays near A1 levels through A2–A4 and jumps only at A5.
4. **Rising Unreliable.** From A1 to A5, Unreliable rises for Qwen2.5-VL (6.6% → 38.8%), GLM (34.4% → 43.5%), Phi-4 (13.3% → 27.6%), and Gemma-4 (2.9% → 8.5%). For Phi-4, the peak is at A4 (39.4%), not A5, and A3 also rises (28.1%).
5. **A5 shifts all six models toward rejection, but in different forms.** Relative to A1, net shift toward rejection at A5 ranges from +5.6 pp (Gemma-4) to +42.2 pp (Qwen3-VL), and the image-only comparison A2→A5 is also positive for all six. The new label differs by model:
   - mainly Uncertain (abstention): Qwen3-VL (3,756 of 4,220 changed detections) and InternVL (1,783 of 2,186);
   - mainly Unreliable: Qwen2.5-VL (1,931 of 2,654), Gemma-4, and Phi-4 (all changes, since they never output Uncertain);
   - both: GLM (1,065 into Uncertain and 1,367 into Unreliable).

   Qwen2.5-VL is the only model whose Uncertain share falls at A5 (30.9% → 7.9%): its A1 Uncertain decisions mostly become Unreliable (1,160 of 1,775).
6. **A4 has no common direction.** Relative to A1, A4 shifts toward rejection for Phi-4 (+26.2 pp) and slightly for Qwen2.5-VL (+3.7) and Gemma-4 (+0.6). It shifts toward acceptance for Qwen3-VL (−14.5), InternVL (−6.0), and GLM (−4.2). The image-only comparison A2→A4 keeps the same direction for every model except InternVL, which flips to +2.6.
7. **Adding the confidence text (A1→A2) shifts all six models toward acceptance.** The size varies widely: Qwen2.5-VL −14.7, Qwen3-VL −9.2, InternVL −8.6, GLM −1.3, Phi-4 −0.5, Gemma-4 −0.2 pp. Adding geometry on top of confidence (A2→A3) changes 3–15% of decisions. It is largest for Phi-4 (15.3%, almost all Reliable→Unreliable) and Qwen3-VL (11.0%).
8. **Nearly invariant model.** Gemma-4 is the only model with fewer than 10% changed detections in every comparison (maximum 8.7%, A1→A5). Every other model changes at least about 20% of decisions in some comparison.

## 6. Interpretation (hedged)

- Removing surrounding context (A5) affects the decisions of every model studied. The models differ mainly in *how* they respond: by abstaining more (Qwen3-VL, InternVL), by rejecting more (Qwen2.5-VL, Phi-4, Gemma-4), or both (GLM). So "context sensitivity" is not one behavior, and a summary measure that merges Uncertain with Unreliable (or drops Uncertain) would hide this difference.
- Adding a crop next to the context (A4) does not act like removing context (A5). For three models it moves decisions in the opposite direction from A5. This is consistent with the two-panel input being processed differently from a crop alone, but the data cannot say which visual cue drives the change.
- The consistent move toward acceptance when confidence text is added (A1→A2) suggests that the numeric YOLO confidence influences the output, even though the prompt calls it auxiliary. The comparison also changes the instruction text (A1 explicitly says not to use confidence), so the prompt wording and the number cannot be separated.
- Models that never output Uncertain (Gemma-4, Phi-4) cannot express abstention in this setup. Their condition sensitivity therefore appears only as Reliable↔Unreliable flips.

## 7. Semantic-performance claims that cannot be made from this analysis

- Whether any condition yields more correct decisions, or whether any model verifies palms better or worse than another.
- Whether moves toward rejection or abstention at A5 are appropriate (for example, correctly rejecting non-palms or correctly flagging ambiguous crops) or inappropriate.
- Whether Gemma-4's invariance reflects robust recognition or insensitivity to the input.
- Anything about precision, recall, specificity, false positives, or any other correctness metric. Those require the semantic ground truth (`docs/SEMANTIC_GT_EVALUATION.md`) and are out of scope here.

**Statistical scope.** These are descriptive counts from a single stored run per model and condition. No significance testing or confidence intervals were computed, and detections are not independent (several come from the same image patch). Repeat-run variability is unknown, so small shifts (for example, Gemma-4 A1→A2, 1.25%) cannot be separated from run-to-run noise without additional runs.

## 8. Validation

Runtime assertions in the script, and the tests in `tests/test_a1_a5_behavior_analysis.py`, check:

- exactly 5,747 detections in every included model-condition;
- identical `sample_id` sets across A1–A5 and against the verification index;
- no duplicate IDs;
- labels restricted to the three allowed values;
- no parse or inference errors;
- `results_index.csv` consistent with the JSONs;
- Table B rows sum to 5,747;
- every transition matrix sums to 5,747;
- unchanged + changed = 5,747;
- toward rejection + toward acceptance = changed, with both equal to their explicit cell definitions;
- transition-matrix margins equal the Table B counts;
- stability counts agree with the per-detection table;
- no ground-truth-derived columns in any output;
- the output directory cannot be inside any protected experiment or review directory.
