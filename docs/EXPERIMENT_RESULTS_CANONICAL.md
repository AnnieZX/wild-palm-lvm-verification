# Experiment Results — Canonical Source of Truth

**Evaluation protocol:** **v2** (current) · **Results date:** 2026-09-27  
**Scope:** This is the single source of truth for **numeric experiment results**. Run inventory, job IDs, checkpoints and qualification history live in [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md). The protocol itself is defined in [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

Every number in this document was regenerated from stored VLM predictions by `scripts/rescore_protocol_v2.py` and read from `outputs/evaluation_protocol_v2/**/A*_metrics.json` (manifest: `outputs/evaluation_protocol_v2/rescore_manifest.csv`). No inference was run. If this document and those artifacts disagree, the artifacts win.

---

## 1. Current evaluation protocol (v2)

| Item | Definition |
|------|------------|
| Detections | **5,747** frozen YOLO detections with confidence ≥ 0.5 (`outputs/verification_dataset/index.csv`) |
| Ground truth | LabelMe shapes whose label satisfies `isinstance(label, str) and label.strip().lower() == "palm"`, converted to the axis-aligned envelope of all shape points (any `shape_type`) |
| Matching | Per image, pairwise IoU, sorted descending, **greedy one-to-one**; a pair matches iff **IoU ≥ 0.5**. Matched detection = GT+, unmatched = GT− |
| Decisions | **Reliable** = positive prediction · **Unreliable** = negative prediction · **Uncertain** = abstention, **excluded** from binary metrics |
| Confusion | TP = GT+ ∧ Reliable · FP = GT− ∧ Reliable · FN = GT+ ∧ Unreliable · TN = GT− ∧ Unreliable |
| Metrics | Accuracy = (TP+TN)/(TP+TN+FP+FN) · Precision = TP/(TP+FP) · Sensitivity = TP/(TP+FN) · **Specificity = TN/(TN+FP)** · F1 · **Balanced Accuracy = (Sensitivity + Specificity)/2** |
| Descriptors | Coverage = (R+Ur)/N · Abstention = U/N (prediction-only, protocol-independent) |

Implementation: `src/preprocessing/gt_palm_bboxes.py` (`is_palm_label`, `EVALUATION_PROTOCOL_VERSION = "v2"`), `src/evaluation/gt_matching.py`, `scripts/evaluate_verification_against_groundtruth.py`, `scripts/compute_verification_metrics.py`.

---

## 2. Protocol v1 → v2 correction

**Why v2 exists.** Protocol v1 selected GT with an exact, case-sensitive `label == "palm"`. 70 of the 880 LabelMe files (parents 0194–0205) label palms as `"Palm"`; their **486** boxes were silently dropped. Detections on real palms in those patches were therefore scored GT− (false positives of YOLO), which penalized every verifier that correctly accepted them.

**What changed.** Only palm-label normalization. IoU threshold, greedy matching, detection set, box envelope, decision semantics, prompts, parser and A1–A5 inputs are unchanged.

| Quantity | Protocol v1 | Protocol v2 |
|----------|------------:|------------:|
| LabelMe GT palm boxes (880 files) | 5,367 | **5,853** |
| Verification GT+ / GT− (N = 5,747) | 4,685 / 1,062 | **5,109 / 638** |
| Always-Reliable accuracy (class prior) | 0.8152 | **0.8890** |
| GT+→GT+ / GT+→GT− / GT−→GT+ / GT−→GT− | — | 4,685 / **0** / **424** / 638 |

The 424 flipped detections lie in 67 images of parents 0194–0205 (`sample_004281`–`sample_004728`). They are clear matches, not borderline ones: corrected IoU 0.512–0.989, median 0.906.

**Why no inference was rerun.** Dataset construction, prompts and A1–A5 images read LabelMe files only for PNG pixels, never for labels. VLM predictions are byte-identical (SHA-256 verified); only GT-derived fields change. For every re-scored run the rescore script verified identical sample IDs, identical R/U/Ur, identical parse/inference failures, and that only `gt_bbox`, `max_iou` and `matched_gt` changed, and only in parents 0194–0205.

**Scope.** All **32** full N = 5,747 condition runs (8 models) change. All @1000, @100, balanced-100, Stage 0, smoke and sanity runs use samples `sample_000001`–`sample_001000` and are **numerically unchanged** (verified for the 28 such runs with evaluation trees).

**Conclusions that changed**

- The class prior rose from 0.815 to **0.889**; accuracy/F1 at the prior now means ≈ 0.89 / ≈ 0.94.
- Specificity rose in 29 of 32 full-scale cells, often substantially (e.g. Qwen2.5-VL A5 0.72 → **0.90**, Qwen3-VL A5 0.81 → **0.95**, Phi-4 A4 0.67 → **0.86**). It fell in two cells (Qwen3-VL A4 0.128 → 0.124, InternVL3.5 A5 0.127 → 0.094), because some former TNs become FNs, and stayed 0 for Molmo2. Accuracy fell for GLM A5 (0.545 → 0.529) and Qwen3-VL A5 (0.715 → 0.710). The full comparison is in §13.
- **InternVL3.5 is no longer "collapsed in every condition".** A1 (Spec 0.366, BalAcc 0.676) and A3 (Spec 0.242, BalAcc 0.620) show moderate specificity at low coverage; A2, A4 and A5 stay near zero (§9).
- The earlier claim that the collapsed models' specificity "never exceeds 0.24" is false: InternVL3.5 A1 reaches 0.366 and Gemma 4 A5 0.332.

**Conclusions that did not change**

- All R/U/Ur decision distributions, coverage and abstention rates.
- The four useful verifiers (Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4) and their qualitative profiles.
- Every useful verifier reaches its highest specificity and its highest balanced accuracy on a crop-based condition (A4 or A5).
- Qwen2.5-VL A3 specificity still collapses (0.038); A5 is still its most conservative condition.
- Reliable-heavy collapse of Gemma 4 and MiniCPM (accuracy still within ±0.011 of the prior); Molmo2 specificity is still 0; LLaVA-OneVision and Gemma 3 are unaffected (@1000).

---

## 3. Dataset statistics

| Asset | Value |
|-------|------:|
| LabelMe files / images | 880 |
| Files labeling palms `palm` / `Palm` / both | 810 / 70 / 0 |
| GT palm boxes (v2) | 5,853 (5,367 `palm` + 486 `Palm`) |
| Palm shape types | 3,306 rotation · 2,544 rectangle · 3 point |
| YOLO detections (all confidences) | 123,405 |
| Verification detections (confidence ≥ 0.5) | 5,747 on 870 images |
| Verification GT+ / GT− | **5,109 / 638** (prior 0.8890) |
| A1@1000 slice (`sample_000001`–`001000`) GT+ / GT− | 928 / 72 (prior 0.928; unchanged) |

**YOLO detection-level metrics (all 123,405 detections, greedy IoU ≥ 0.5)** — `outputs/evaluation_protocol_v2/detection_metrics.json`:

| | GT palms | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2** | 5,853 | 5,810 | 117,595 | 43 | 0.0471 | 0.9927 | 0.0899 |
| v1 (provenance) | 5,367 | 5,329 | 118,076 | 38 | 0.0432 | 0.9929 | 0.0828 |

Three of the 43 v2 false negatives are zero-area point annotations that can never be matched (§12).

---

## 4. Model status summary

| Model | Furthest evaluation | Run status | Outcome (v2) |
|-------|---------------------|-----------|--------------|
| Qwen2.5-VL-7B-Instruct | A1–A5 @5747 | Complete | Useful verifier |
| Qwen3-VL-8B-Instruct | A1–A5 @5747 | Complete | Useful verifier |
| GLM-4.6V-Flash | A1–A5 @5747 | Complete | Useful verifier |
| Phi-4-multimodal-instruct | A1–A5 @5747 | Complete | Useful verifier (binary; never Uncertain) |
| InternVL3.5-8B-HF | A1–A5 @5747 | Complete | Abstention-heavy; moderate specificity on A1 and A3 at low coverage, near-zero on A2, A4, A5 |
| Gemma 4 12B IT | A1–A5 @5747 | Complete | Reliable-heavy collapse (gate deviation) |
| MiniCPM-V-4.5 | A1 @5747 | A1 only | Reliable-heavy collapse |
| Molmo2-8B | A1 @5747 | A1 only | Abstention-heavy collapse |
| LLaVA-OneVision | A1 @1000 | A1 only | Single-class Reliable collapse |
| Gemma 3 12B IT | A1 @1000 | A1 only | Single-class Reliable collapse |
| InternVL3-8B-Instruct | Balanced-100 | Qualification failed / stopped | Technical failure |
| Llama-3.2-11B-Vision-Instruct | Engineering Stage 0 | Integrated; engineering qualification only | No scientific result |
| Ministral-3-8B | — | Integrated; not evaluated | — |

Details, job IDs and gate history: [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md).

---

## 5. Full A1–A5 results (N = 5,747, Protocol v2)

Binary metrics exclude Uncertain. Parse and inference failures are counted from the stored prediction JSONs.

| Model | Cond | N | R | U | Ur | Acc | Prec | Sens | Spec | F1 | BalAcc | TP | TN | FP | FN | Parse fail | Infer fail |
|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **Qwen2.5-VL-7B** | A1 | 5747 | 3593 | 1775 | 379 | 0.8935 | 0.9544 | 0.9298 | 0.4225 | 0.9419 | 0.6762 | 3429 | 120 | 164 | 259 | 0 | 0 |
|  | A2 | 5747 | 4268 | 1344 | 135 | 0.9280 | 0.9456 | 0.9794 | 0.1773 | 0.9622 | 0.5783 | 4036 | 50 | 232 | 85 | 0 | 0 |
|  | A3 | 5747 | 4416 | 1313 | 18 | 0.9407 | 0.9423 | 0.9981 | 0.0377 | 0.9694 | 0.5179 | 4161 | 10 | 255 | 8 | 0 | 0 |
|  | A4 | 5747 | 3570 | 1557 | 620 | 0.8716 | 0.9692 | 0.8899 | 0.6358 | 0.9279 | 0.7628 | 3460 | 192 | 110 | 428 | 0 | 0 |
|  | A5 | 5747 | 3065 | 455 | 2227 | 0.6674 | 0.9811 | 0.6386 | 0.9005 | 0.7736 | 0.7695 | 3007 | 525 | 58 | 1702 | 0 | 0 |
| **Qwen3-VL-8B** | A1 | 5747 | 4309 | 246 | 1192 | 0.7931 | 0.9371 | 0.8232 | 0.5453 | 0.8765 | 0.6843 | 4038 | 325 | 271 | 867 | 0 | 0 |
|  | A2 | 5747 | 4636 | 401 | 710 | 0.8444 | 0.9277 | 0.8964 | 0.3887 | 0.9118 | 0.6426 | 4301 | 213 | 335 | 497 | 0 | 0 |
|  | A3 | 5747 | 4221 | 367 | 1159 | 0.7931 | 0.9377 | 0.8232 | 0.5402 | 0.8767 | 0.6817 | 3958 | 309 | 263 | 850 | 0 | 0 |
|  | A4 | 5747 | 5051 | 276 | 420 | 0.8424 | 0.8980 | 0.9289 | 0.1241 | 0.9132 | 0.5265 | 4536 | 73 | 515 | 347 | 0 | 0 |
|  | A5 | 5747 | 1081 | 3948 | 718 | 0.7098 | 0.9898 | 0.6768 | 0.9495 | 0.8039 | 0.8132 | 1070 | 207 | 11 | 511 | 0 | 0 |
| **GLM-4.6V-Flash** | A1 | 5747 | 3719 | 50 | 1978 | 0.6874 | 0.9419 | 0.6912 | 0.6566 | 0.7973 | 0.6739 | 3503 | 413 | 216 | 1565 | 0 | 0 |
|  | A2 | 5747 | 3783 | 57 | 1907 | 0.7039 | 0.9471 | 0.7070 | 0.6785 | 0.8096 | 0.6927 | 3583 | 422 | 200 | 1485 | 0 | 0 |
|  | A3 | 5747 | 3696 | 100 | 1951 | 0.7053 | 0.9562 | 0.7017 | 0.7349 | 0.8094 | 0.7183 | 3534 | 449 | 162 | 1502 | 0 | 0 |
|  | A4 | 5747 | 3960 | 64 | 1723 | 0.7406 | 0.9530 | 0.7456 | 0.7005 | 0.8366 | 0.7230 | 3774 | 435 | 186 | 1288 | 0 | 0 |
|  | A5 | 5747 | 2163 | 1082 | 2502 | 0.5286 | 0.9838 | 0.4958 | 0.9062 | 0.6593 | 0.7010 | 2128 | 338 | 35 | 2164 | 0 | 0 |
| **Phi-4 Multimodal** | A1 | 5747 | 4985 | 0 | 762 | 0.8135 | 0.9049 | 0.8830 | 0.2571 | 0.8938 | 0.5700 | 4511 | 164 | 474 | 598 | 0 | 0 |
|  | A2 | 5747 | 5012 | 0 | 735 | 0.8227 | 0.9080 | 0.8908 | 0.2774 | 0.8993 | 0.5841 | 4551 | 177 | 461 | 558 | 0 | 0 |
|  | A3 | 5747 | 4132 | 0 | 1615 | 0.7221 | 0.9250 | 0.7481 | 0.5141 | 0.8272 | 0.6311 | 3822 | 328 | 310 | 1287 | 0 | 0 |
|  | A4 | 5747 | 3481 | 0 | 2266 | 0.6851 | 0.9739 | 0.6635 | 0.8574 | 0.7893 | 0.7605 | 3390 | 547 | 91 | 1719 | 0 | 0 |
|  | A5 | 5747 | 4160 | 0 | 1587 | 0.7674 | 0.9534 | 0.7763 | 0.6959 | 0.8558 | 0.7361 | 3966 | 444 | 194 | 1143 | 0 | 0 |
| **InternVL3.5-8B-HF** | A1 | 5747 | 3085 | 2570 | 92 | 0.9597 | 0.9724 | 0.9859 | 0.3657 | 0.9791 | 0.6758 | 3000 | 49 | 85 | 43 | 0 | 0 |
|  | A2 | 5747 | 3497 | 2239 | 11 | 0.9778 | 0.9783 | 0.9994 | 0.1059 | 0.9887 | 0.5526 | 3421 | 9 | 76 | 2 | 0 | 0 |
|  | A3 | 5747 | 3420 | 2298 | 29 | 0.9780 | 0.9798 | 0.9979 | 0.2418 | 0.9888 | 0.6198 | 3351 | 22 | 69 | 7 | 0 | 0 |
|  | A4 | 5747 | 3343 | 2399 | 5 | 0.9743 | 0.9743 | 1.0000 | 0.0549 | 0.9870 | 0.5275 | 3257 | 5 | 86 | 0 | 0 | 0 |
|  | A5 | 5747 | 1515 | 4078 | 154 | 0.8922 | 0.9809 | 0.9078 | 0.0938 | 0.9429 | 0.5008 | 1486 | 3 | 29 | 151 | 0 | 0 |
| **Gemma 4 12B IT** | A1 | 5747 | 5583 | 0 | 164 | 0.8834 | 0.8975 | 0.9808 | 0.1034 | 0.9373 | 0.5421 | 5011 | 66 | 572 | 98 | 0 | 0 |
|  | A2 | 5747 | 5595 | 0 | 152 | 0.8827 | 0.8963 | 0.9816 | 0.0909 | 0.9370 | 0.5363 | 5015 | 58 | 580 | 94 | 0 | 0 |
|  | A3 | 5747 | 5424 | 0 | 323 | 0.8843 | 0.9097 | 0.9657 | 0.2320 | 0.9369 | 0.5989 | 4934 | 148 | 490 | 175 | 0 | 0 |
|  | A4 | 5747 | 5547 | 0 | 200 | 0.8952 | 0.9063 | 0.9839 | 0.1850 | 0.9435 | 0.5845 | 5027 | 118 | 520 | 82 | 0 | 0 |
|  | A5 | 5747 | 5261 | 0 | 486 | 0.8782 | 0.9190 | 0.9464 | 0.3323 | 0.9325 | 0.6393 | 4835 | 212 | 426 | 274 | 0 | 0 |

> **Accuracy and F1 are not verification skill on their own.** The always-Reliable policy scores Acc 0.889 / F1 0.941 on this set. InternVL3.5's Acc/F1 are the highest in the table only because 39–71% of its samples are Uncertain and excluded, and it almost never answers Unreliable. Gemma 4's accuracy (0.878–0.895) sits at the prior. Read Specificity, Balanced Accuracy, coverage and the R/U/Ur mix together.

---

## 6. A1-only results @5747 (A2–A5 not evaluated)

| Model | Cond | N | R | U | Ur | Acc | Prec | Sens | Spec | F1 | BalAcc | TP | TN | FP | FN | Parse fail | Infer fail |
|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **MiniCPM-V-4.5** | A1 | 5747 | 5557 | 0 | 190 | 0.8810 | 0.8981 | 0.9769 | 0.1129 | 0.9359 | 0.5449 | 4991 | 72 | 566 | 118 | 0 | 0 |
| **Molmo2-8B** | A1 | 5747 | 1209 | 4537 | 1 | 0.9488 | 0.9495 | 0.9991 | 0.0000 | 0.9737 | 0.4996 | 1148 | 0 | 61 | 1 | 0 | 0 |

Molmo2's F1 is computed on only 1,210 scored samples (coverage 0.21), among which it rejected a single box.

---

## 7. Unchanged sub-5,747 results (@1000, @100, smoke)

These runs use only `sample_000001`–`sample_001000` (A1@1000 slice prior 0.928; the Gemma 4 pre-check uses the first 100 of that slice). Their binary metrics are identical under v1 and v2 (verified by `scripts/rescore_protocol_v2.py`). Do **not** compare them directly with full-set numbers.

| Model | Experiment | Cond | N | R / U / Ur | Acc | Prec | Sens | Spec | F1 | BalAcc | TP/TN/FP/FN |
|---|---|:-:|--:|---|--:|--:|--:|--:|--:|--:|---|
| Qwen2.5-VL-7B | `20260706_2214` | A1 | 1000 | 672 / 272 / 56 | 0.8997 | 0.9628 | 0.9309 | 0.2424 | 0.9466 | 0.5867 | 647/8/25/48 |
| | | A2 | 1000 | 768 / 211 / 21 | 0.9379 | 0.9609 | 0.9749 | 0.0625 | 0.9679 | 0.5187 | 738/2/30/19 |
| | | A3 | 1000 | 799 / 200 / 1 | 0.9537 | 0.9549 | 0.9987 | 0.0000 | 0.9763 | 0.4993 | 763/0/36/1 |
| | | A4 | 1000 | 672 / 253 / 75 | 0.8929 | 0.9717 | 0.9146 | 0.4242 | 0.9423 | 0.6694 | 653/14/19/61 |
| | | A5 | 1000 | 571 / 89 / 340 | 0.6762 | 0.9825 | 0.6631 | 0.8462 | 0.7918 | 0.7546 | 561/55/10/285 |
| Qwen3-VL-8B | `qwen3vl_A1_1000` | A1 | 1000 | 724 / 48 / 228 | 0.7616 | 0.9558 | 0.7802 | 0.5077 | 0.8591 | 0.6439 | 692/33/32/195 |
| GLM-4.6V-Flash | `20260913_glm46v_flash_A1_1000` | A1 | 1000 | 641 / 8 / 351 | 0.6583 | 0.9532 | 0.6641 | 0.5833 | 0.7828 | 0.6237 | 611/42/30/309 |
| | `20260914_glm46v_flash_A2A5_1000` | A2 | 1000 | 650 / 7 / 343 | 0.6667 | 0.9538 | 0.6732 | 0.5833 | 0.7893 | 0.6283 | 620/42/30/301 |
| | | A3 | 1000 | 638 / 11 / 351 | 0.6764 | 0.9687 | 0.6732 | 0.7183 | 0.7943 | 0.6958 | 618/51/20/300 |
| | | A4 | 1000 | 676 / 7 / 317 | 0.7069 | 0.9660 | 0.7090 | 0.6806 | 0.8178 | 0.6948 | 653/49/23/268 |
| | | A5 | 1000 | 400 / 179 / 421 | 0.5213 | 0.9800 | 0.5045 | 0.8182 | 0.6661 | 0.6613 | 392/36/8/385 |
| Phi-4 Multimodal | `20260919_1524_phi4_A1A5_1000` | A1 | 1000 | 910 / 0 / 90 | 0.8540 | 0.9297 | 0.9116 | 0.1111 | 0.9206 | 0.5114 | 846/8/64/82 |
| | `20260920_2339_phi4_A2_1000` | A2 | 1000 | 909 / 0 / 91 | 0.8570 | 0.9318 | 0.9127 | 0.1389 | 0.9222 | 0.5258 | 847/10/62/81 |
| | `20260919_1524_phi4_A1A5_1000` | A3 | 1000 | 793 / 0 / 207 | 0.7670 | 0.9382 | 0.8017 | 0.3194 | 0.8646 | 0.5606 | 744/23/49/184 |
| | | A4 | 1000 | 652 / 0 / 348 | 0.6880 | 0.9724 | 0.6832 | 0.7500 | 0.8025 | 0.7166 | 634/54/18/294 |
| | | A5 | 1000 | 762 / 0 / 238 | 0.7700 | 0.9580 | 0.7866 | 0.5556 | 0.8639 | 0.6711 | 730/40/32/198 |
| MiniCPM-V-4.5 | `minicpm_A1_1000` | A1 | 1000 | 957 / 0 / 43 | 0.9030 | 0.9342 | 0.9634 | 0.1250 | 0.9485 | 0.5442 | 894/9/63/34 |
| Molmo2-8B | `molmo2_A1_1000` | A1 | 1000 | 198 / 801 / 1 | 0.9598 | 0.9646 | 0.9948 | 0.0000 | 0.9795 | 0.4974 | 191/0/7/1 |
| LLaVA-OneVision | `20260719_1734` | A1 | 1000 | 1000 / 0 / 0 | 0.9280 | 0.9280 | 1.0000 | 0.0000 | 0.9627 | 0.5000 | 928/0/72/0 |
| Gemma 3 12B IT | `20260802_1702` | A1 | 1000 | 1000 / 0 / 0 | 0.9280 | 0.9280 | 1.0000 | 0.0000 | 0.9627 | 0.5000 | 928/0/72/0 |
| Gemma 4 12B IT | `20260925_gemma4_qual` | A1 | 100 | 95 / 0 / 5 | 0.9300 | 0.9474 | 0.9783 | 0.3750 | 0.9626 | 0.6766 | 90/3/5/2 |

Smoke/sanity runs (≤ 20 samples: MiniCPM, Molmo2, Phi-4, Qwen3-VL, Qwen2.5-VL, Qwen A5′ smoke) are engineering checks and are not scientific results; their re-scored artifacts are in the v2 tree.

**Qualification gates (unchanged; drawn from the A1@1000 slice):**

| Model | Gate | R / U / Ur | Parse fail | Spec | BalAcc | Verdict |
|---|---|---|--:|--:|--:|---|
| GLM-4.6V-Flash | Balanced-100 (`20260913_glm46v_flash_stage1_balanced100`) | 52 / 2 / 46 | 0 | 0.56 | 0.59 | Pass |
| InternVL3.5-8B-HF | Balanced-100 (`20260924_internvl3_5_hf_qual`) | 39 / 58 / 3 | 0 | 0.20 | 0.60 | Pass |
| InternVL3-8B-Instruct | Balanced-100 (`20260909_internvl3_qual`) | 56 / 31 / 0 | 13 | 0.00 | 0.50 | Fail (technical) |

---

## 8. Qualification failures and engineering-only runs

- **InternVL3-8B-Instruct** — balanced-100 FAIL_TECHNICAL: 13/100 unparsable replies, Spec 0.00, BalAcc 0.50. Stopped; never run at scale. See [`INTERNVL3_QUALIFICATION.md`](INTERNVL3_QUALIFICATION.md).
- **Llama-3.2-11B-Vision-Instruct** — integrated (config, adapter, jobs). Only an engineering Stage 0 run exists (`20260927_llama3_2_11b_vision_engineering_qualification`, 10 samples: 9 Reliable, 1 parse failure). This validates the integration only and is **not** a scientific result.
- **Ministral-3-8B** — integrated (config, adapter, jobs); no experiment run.

---

## 9. Model behavior and outcome classifications

Outcome labels describe the observed decision mix; they are not a pass/fail gate and do not change the evaluator.

| Class | Models | Evidence (v2) |
|---|---|---|
| **Useful verifier** | Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4 | Use rejection with substantial specificity in at least one condition (best Spec 0.90 / 0.95 / 0.91 / 0.86; best BalAcc 0.770 / 0.813 / 0.723 / 0.760) |
| **Abstention-heavy, condition-dependent** | InternVL3.5-8B-HF | Uncertain 39–71%, Unreliable ≤ 2.7% in every condition. Moderate specificity on A1 (0.366, BalAcc 0.676, coverage 0.55) and A3 (0.242, BalAcc 0.620, coverage 0.60); near-zero on A2 (0.106), A4 (0.055), A5 (0.094). High Acc/F1 are computed on the non-Uncertain subset |
| **Reliable-heavy collapse** | Gemma 4 12B IT, MiniCPM-V-4.5 | Reliable 91.5–97.4% (Gemma 4) and 96.7% (MiniCPM), zero Uncertain; accuracy 0.878–0.895 and 0.881 vs prior 0.889; Spec 0.091–0.332 and 0.113 |
| **Abstention-heavy collapse** | Molmo2-8B | 79% Uncertain, one Unreliable in 5,747; Spec 0.000 |
| **Single-class Reliable collapse** | LLaVA-OneVision, Gemma 3 | 1000/0/0 at A1@1000; Acc = slice prior 0.928; Spec 0 |
| **Technical failure** | InternVL3-8B-Instruct | 13% parse failures at the qualification gate |

InternVL3.5 is deliberately **not** grouped with the Reliable-output collapse models: it abstains rather than accepts, and two of its conditions show non-trivial specificity on the samples it does decide.

---

## 10. A1–A5 findings (full scale, v2)

1. **Crop-based context gives the best rejection.** Every useful verifier reaches its highest specificity on a crop-based condition — Qwen2.5-VL A5 (0.901), Qwen3-VL A5 (0.950), GLM A5 (0.906), Phi-4 A4 (0.857) — and its highest balanced accuracy on A4 or A5 (Qwen2.5-VL A5 0.770, Qwen3-VL A5 0.813, GLM A4 0.723, Phi-4 A4 0.760). This usually costs sensitivity.
2. **No condition is best for everything.** Qwen2.5-VL has its highest F1 and sensitivity on A3 (0.969 / 0.998) but its lowest specificity there (0.038). The A5 vs A4 balanced-accuracy lead for Qwen2.5-VL is narrow (0.770 vs 0.763).
3. **Adding YOLO confidence to the overlay (A1 → A2) lowers specificity for the Qwen family** (Qwen2.5-VL 0.42 → 0.18, Qwen3-VL 0.55 → 0.39) while slightly raising it for GLM (0.66 → 0.68) and Phi-4 (0.26 → 0.28). Adding geometry (A3) collapses Qwen2.5-VL specificity to 0.038.
4. **GLM-4.6V-Flash has the most consistent specificity** across conditions (0.657–0.906), with the lowest sensitivity (0.50–0.75).
5. **Qwen3-VL A5 is the most conservative cell**: Spec 0.950 and precision 0.990, but it abstains on 69% of boxes (coverage 0.31).

---

## 11. Prediction-distribution findings (protocol-independent)

| Model | R% range (A1–A5) | U% range | Ur% range | Coverage range |
|---|---|---|---|---|
| Qwen2.5-VL | 53.3–76.8 | 7.9–30.9 | 0.3–38.8 | 0.69–0.92 |
| Qwen3-VL | 18.8–87.9 | 4.3–68.7 | 7.3–20.7 | 0.31–0.96 |
| GLM-4.6V-Flash | 37.6–68.9 | 0.9–18.8 | 30.0–43.5 | 0.81–0.99 |
| Phi-4 | 60.6–87.2 | 0 | 12.8–39.4 | 1.00 |
| InternVL3.5 | 26.4–60.9 | 39.0–71.0 | 0.1–2.7 | 0.29–0.61 |
| Gemma 4 | 91.5–97.4 | 0 | 2.6–8.5 | 1.00 |
| MiniCPM (A1) | 96.7 | 0 | 3.3 | 1.00 |
| Molmo2 (A1) | 21.0 | 79.0 | 0.02 | 0.21 |

- Qwen2.5-VL uses all three verdicts; Qwen3-VL abstains heavily only on A5; GLM rejects about a third of boxes; Phi-4 never answers Uncertain.
- The same model's decision mix moves strongly with context (Qwen3-VL Uncertain 4% on A1 → 69% on A5).

---

## 12. Known limitations and data-quality issues

| Issue | Classification | Effect |
|---|---|---|
| Case-sensitive palm label (Protocol v1) | Protocol bug — **fixed in v2** | 486 boxes, 424 detection labels |
| 25 detections with IoU ≥ 0.5 that lose greedy one-to-one matching (24 under v1) | Expected protocol behavior (VOC/COCO) | Near-duplicate YOLO boxes of an already-matched palm are GT−; a verifier answering Reliable on them scores an FP |
| 3 point-shaped `palm` annotations: `100_0003_0028_2` (6.60, 695.16), `100_0003_0032_3` (282.51, 271.06), `100_0003_0350_8` (5.40, 602.39) | Data-quality issue (kept in v2) | Zero-area GT boxes, never matchable; add 3 detection-level FNs; **no effect** on the 5,747 verification labels |
| Rotated palm shapes (3,306) scored by their axis-aligned envelope | Expected protocol behavior | Same in v1 and v2 |
| 744 of 5,853 GT palms have no matched detection with confidence ≥ 0.5 | Expected | Detector recall; not part of verification metrics |
| Duplicate GT boxes, malformed shapes, empty or missing JSONs, image/annotation size mismatches | None found | — |
| Gemma 4 skipped the qualification ladder | Documented deviation | Valid observation, not a gated one |
| `reliable_pct` etc. in sub-5,747 metrics JSONs are divided by 5,747 | Evaluator reporting quirk | Use raw counts for @1000/@100 runs |

---

## 13. Protocol v1 provenance

Protocol v1 (`label == "palm"`, GT 4,685 / 1,062, prior 0.8152) was the evaluation protocol from the first evaluation code (July 4–5, 2026) until 2026-09-27. Its outputs are frozen and unmodified in `outputs/evaluation/` (and `outputs/evaluation/detection_metrics.json`). Archived documents under [`archive/docs/`](../archive/docs/README.md) report v1 numbers and are labeled as such.

Protocol v1 → v2 specificity for the 32 full-scale cells. v1 accuracy, precision, recall, F1 and confusion counts are in `outputs/evaluation_protocol_v2/rescore_manifest.csv` (columns `v1_*`); v1 specificity = `v1_true_negative / (v1_true_negative + v1_false_positive)`.

| Model | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Qwen2.5-VL | 0.3059 → 0.4225 | 0.1045 → 0.1773 | 0.0179 → 0.0377 | 0.4327 → 0.6358 | 0.7198 → 0.9005 |
| Qwen3-VL | 0.4138 → 0.5453 | 0.2857 → 0.3887 | 0.4130 → 0.5402 | 0.1275 → 0.1241 | 0.8121 → 0.9495 |
| GLM-4.6V-Flash | 0.5492 → 0.6566 | 0.5547 → 0.6785 | 0.5912 → 0.7349 | 0.5592 → 0.7005 | 0.7607 → 0.9062 |
| Phi-4 | 0.2260 → 0.2571 | 0.2335 → 0.2774 | 0.4369 → 0.5141 | 0.6733 → 0.8574 | 0.5273 → 0.6959 |
| InternVL3.5 | 0.1436 → 0.3657 | 0.0265 → 0.1059 | 0.0651 → 0.2418 | 0.0157 → 0.0549 | 0.1269 → 0.0938 |
| Gemma 4 | 0.0782 → 0.1034 | 0.0678 → 0.0909 | 0.1657 → 0.2320 | 0.1215 → 0.1850 | 0.2363 → 0.3323 |
| MiniCPM (A1) | 0.0782 → 0.1129 | | | | |
| Molmo2 (A1) | 0.0000 → 0.0000 | | | | |

Timeline: case-sensitive palm filtering first appears in commit `9ae7126` (2026-06-14, input tooling); the evaluation filter was introduced in `5f7fadb` (2026-07-04) and moved to `src/preprocessing/gt_palm_bboxes.py` in `d632e7a` (2026-07-05), unchanged until Protocol v2. The first affected results were the YOLO detection metrics (2026-07-05) and Qwen2.5-VL `20260708_0020`. The defect was first documented on 2026-09-27. The `"Palm"` annotation files live outside git (`/deac/csc/yangGrp/cuij/palm/Raw_Patches`), so their creation date cannot be established from repository evidence.

---

## 14. Experiment IDs and artifact locations

| Model | Registry key | Full / largest experiment | Predictions | v2 evaluation |
|---|---|---|---|---|
| Qwen2.5-VL-7B | `qwen2_5_vl` (legacy tree `qwen`) | `20260708_0020` | `outputs/verification/qwen/20260708_0020/` | `outputs/evaluation_protocol_v2/qwen/20260708_0020/` |
| Qwen3-VL-8B | `qwen3_vl` | `qwen3vl_A1A5_5747` | `outputs/verification/qwen3_vl/…` | `outputs/evaluation_protocol_v2/qwen3_vl/…` |
| GLM-4.6V-Flash | `glm_4_6v_flash` | `20260919_glm46v_flash_A1A5_5747` | `outputs/verification/glm_4_6v_flash/…` | `outputs/evaluation_protocol_v2/glm_4_6v_flash/…` |
| Phi-4 Multimodal | `phi4_multimodal` | `20260921_phi4_A1A5_5747` | `outputs/verification/phi4_multimodal/…` | `outputs/evaluation_protocol_v2/phi4_multimodal/…` |
| InternVL3.5-8B-HF | `internvl3_5_hf` | `20260924_internvl3_5_hf_A1A5_5747` | `outputs/verification/internvl3_5_hf/…` | `outputs/evaluation_protocol_v2/internvl3_5_hf/…` |
| Gemma 4 12B IT | `gemma4` | `20260925_gemma4_A1A5_5747` | `outputs/verification/gemma4/…` | `outputs/evaluation_protocol_v2/gemma4/…` |
| MiniCPM-V-4.5 | `minicpm_v4_5` | `20260923_minicpm_A1A5_5747` (A1) | `outputs/verification/minicpm_v4_5/…` | `outputs/evaluation_protocol_v2/minicpm_v4_5/…` |
| Molmo2-8B | `molmo2_8b` | `20260923_molmo2_A1A5_5747` (A1) | `outputs/verification/molmo2_8b/…` | `outputs/evaluation_protocol_v2/molmo2_8b/…` |
| LLaVA-OneVision | `llava` | `20260719_1734` (A1@1000) | `outputs/verification/llava/…` | `outputs/evaluation_protocol_v2/llava/…` |
| Gemma 3 12B IT | `gemma` | `20260802_1702` (A1@1000) | `outputs/verification/gemma/…` | `outputs/evaluation_protocol_v2/gemma/…` |

Each v2 experiment directory contains `PROTOCOL.json` (protocol version, GT rule, IoU threshold, matching, generation time, base commit). The v2 root holds `PROTOCOL.json`, `rescore_manifest.csv` and `detection_metrics.json`. Protocol v1 twins stay at `outputs/evaluation/<same relative path>`. All `outputs/` trees are gitignored on-disk evidence.
