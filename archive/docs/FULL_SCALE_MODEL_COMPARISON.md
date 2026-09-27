> **ARCHIVED 2026-09-27 — historical snapshot, not current.** Any numeric results here use **Evaluation Protocol v1** (case-sensitive `"palm"` GT label; GT+ 4,685 / GT− 1,062) and are superseded. Current results: [`docs/EXPERIMENT_RESULTS_CANONICAL.md`](../../docs/EXPERIMENT_RESULTS_CANONICAL.md) · current status: [`docs/EXPERIMENT_STATUS_CANONICAL.md`](../../docs/EXPERIMENT_STATUS_CANONICAL.md) · index: [`archive/docs/README.md`](README.md).

# Full-Scale Model Comparison (N = 5,747)

**Status date:** 2026-09-27 (canonicalization; all numbers re-derived from raw prediction + evaluation trees)  
**Inventory, job IDs, qualification path, reproducibility:** [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md)

This document is the **advisor-facing** summary of the frozen A1–A5 verification benchmark. It does **not** rank models or declare a winner.

**Two separate fields:** *Run status* (Complete · A1 only · Qualification failed / stopped · Not evaluated) and *Scientific outcome* (Useful verifier · Reliable-heavy collapse · Abstention-heavy collapse · Technical failure). See canonical §0.

---

## A. Experiment protocol

| Item | Definition |
|------|------------|
| **Task** | Second-stage VLM verification of fixed YOLO palm detections |
| **N** | **5,747** detections (YOLO confidence ≥ 0.5; full production set) |
| **GT** | LabelMe `palm` boxes; pairwise IoU, greedy one-to-one, **IoU ≥ 0.5** |
| **GT prior** | GT+ = 4,685 · GT− = 1,062 · always-Reliable Acc = **0.8152** |
| **Decisions** | **Reliable** (accept) · **Uncertain** (abstain) · **Unreliable** (reject) |
| **Binary metrics** | Reliable = positive; Unreliable = negative; **Uncertain excluded** from Acc / Prec / Sens / Spec / F1 |
| **Ablations** | Frozen **A1–A5** (same boxes, parser, evaluator; only VLM inputs change) |

| Condition | Image input | Metadata in prompt |
|-----------|-------------|--------------------|
| **A1** | Overlay only | None |
| **A2** | Overlay | YOLO confidence |
| **A3** | Overlay | Confidence + bbox geometry |
| **A4** | Dual panel (overlay + crop) | YOLO confidence |
| **A5** | Crop only | YOLO confidence |

Derived columns (documentation only; not written by the frozen evaluator): Specificity = TN/(TN+FP); BalAcc = (Sens+Spec)/2; Coverage = (R+Ur)/N.

All cells in sections C–D are **Complete** and verified (5747 unique sample IDs per condition, matching evaluation, **0 parse / 0 inference errors**).

---

## B. Main status table

| Model | A1 | A2 | A3 | A4 | A5 | Run status | Scientific outcome |
|-------|----|----|----|----|----|-----------|--------------------|
| **Qwen2.5-VL-7B** | Complete | Complete | Complete | Complete | Complete | **Complete** | Useful verifier |
| **Qwen3-VL-8B** | Complete | Complete | Complete | Complete | Complete | **Complete** | Useful verifier |
| **GLM-4.6V-Flash** | Complete | Complete | Complete | Complete | Complete | **Complete** | Useful verifier |
| **Phi-4 Multimodal** | Complete | Complete | Complete | Complete | Complete | **Complete** | Useful verifier (0 Uncertain) |
| **InternVL3.5-8B-HF** | Complete | Complete | Complete | Complete | Complete | **Complete** | Abstention-heavy collapse |
| **Gemma 4 12B IT** | Complete | Complete | Complete | Complete | Complete | **Complete** | Reliable-heavy collapse (gate deviation — canonical §6) |
| MiniCPM-V-4.5 | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated | **A1 only** | Reliable-heavy collapse (§F) |
| Molmo2-8B | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated | **A1 only** | Abstention-heavy collapse (§F) |

---

## C. Main metrics — six fully evaluated models

Uncertain predictions are **not** in the binary denominators.

| Model | Cond | Accuracy | Precision | Sensitivity | Specificity | F1 | BalAcc | Coverage |
|-------|------|---------:|----------:|------------:|------------:|---:|-------:|---------:|
| Qwen2.5-VL | A1 | 0.8512 | 0.8945 | 0.9381 | 0.3059 | 0.9158 | 0.6220 | 0.6911 |
| Qwen2.5-VL | A2 | 0.8662 | 0.8796 | 0.9804 | 0.1045 | 0.9273 | 0.5425 | 0.7661 |
| Qwen2.5-VL | A3 | 0.8742 | 0.8755 | 0.9979 | 0.0179 | 0.9327 | 0.5079 | 0.7715 |
| Qwen2.5-VL | A4 | 0.8332 | 0.9067 | 0.8984 | 0.4327 | 0.9026 | 0.6656 | 0.7291 |
| Qwen2.5-VL | A5 | 0.6604 | 0.9106 | 0.6470 | 0.7198 | 0.7565 | 0.6834 | 0.9208 |
| Qwen3-VL | A1 | 0.7519 | 0.8635 | 0.8273 | 0.4138 | 0.8450 | 0.6205 | 0.9572 |
| Qwen3-VL | A2 | 0.7920 | 0.8555 | 0.8997 | 0.2857 | 0.8770 | 0.5927 | 0.9302 |
| Qwen3-VL | A3 | 0.7532 | 0.8650 | 0.8281 | 0.4130 | 0.8461 | 0.6205 | 0.9361 |
| Qwen3-VL | A4 | 0.7887 | 0.8293 | 0.9344 | 0.1275 | 0.8787 | 0.5310 | 0.9520 |
| Qwen3-VL | A5 | 0.7154 | 0.9426 | 0.6937 | 0.8121 | 0.7992 | 0.7529 | 0.3130 |
| GLM-4.6V-Flash | A1 | 0.6709 | 0.8731 | 0.6983 | 0.5492 | 0.7760 | 0.6237 | 0.9913 |
| GLM-4.6V-Flash | A2 | 0.6849 | 0.8773 | 0.7141 | 0.5547 | 0.7873 | 0.6344 | 0.9901 |
| GLM-4.6V-Flash | A3 | 0.6876 | 0.8866 | 0.7090 | 0.5912 | 0.7879 | 0.6501 | 0.9826 |
| GLM-4.6V-Flash | A4 | 0.7185 | 0.8843 | 0.7541 | 0.5592 | 0.8140 | 0.6566 | 0.9889 |
| GLM-4.6V-Flash | A5 | 0.5445 | 0.9200 | 0.5048 | 0.7607 | 0.6519 | 0.6328 | 0.8117 |
| Phi-4 Multimodal | A1 | 0.7661 | 0.8351 | 0.8886 | 0.2260 | 0.8610 | 0.5573 | 1.0000 |
| Phi-4 Multimodal | A2 | 0.7736 | 0.8376 | 0.8961 | 0.2335 | 0.8658 | 0.5648 | 1.0000 |
| Phi-4 Multimodal | A3 | 0.6957 | 0.8553 | 0.7543 | 0.4369 | 0.8016 | 0.5956 | 1.0000 |
| Phi-4 Multimodal | A4 | 0.6697 | 0.9003 | 0.6689 | 0.6733 | 0.7676 | 0.6711 | 1.0000 |
| Phi-4 Multimodal | A5 | 0.7339 | 0.8793 | 0.7808 | 0.5273 | 0.8271 | 0.6540 | 1.0000 |
| InternVL3.5-8B-HF | A1 | 0.8898 | 0.8995 | 0.9858 | 0.1436 | 0.9407 | 0.5647 | 0.5528 |
| InternVL3.5-8B-HF | A2 | 0.9051 | 0.9053 | 0.9994 | 0.0265 | 0.9500 | 0.5129 | 0.6104 |
| InternVL3.5-8B-HF | A3 | 0.9063 | 0.9076 | 0.9977 | 0.0651 | 0.9505 | 0.5314 | 0.6001 |
| InternVL3.5-8B-HF | A4 | 0.9065 | 0.9064 | 1.0000 | 0.0157 | 0.9509 | 0.5079 | 0.5826 |
| InternVL3.5-8B-HF | A5 | 0.8478 | 0.9228 | 0.9107 | 0.1269 | 0.9167 | 0.5188 | 0.2904 |
| Gemma 4 12B IT | A1 | 0.8156 | 0.8246 | 0.9827 | 0.0782 | 0.8968 | 0.5304 | 1.0000 |
| Gemma 4 12B IT | A2 | 0.8138 | 0.8231 | 0.9829 | 0.0678 | 0.8959 | 0.5254 | 1.0000 |
| Gemma 4 12B IT | A3 | 0.8203 | 0.8367 | 0.9686 | 0.1657 | 0.8978 | 0.5672 | 1.0000 |
| Gemma 4 12B IT | A4 | 0.8253 | 0.8318 | 0.9848 | 0.1215 | 0.9019 | 0.5532 | 1.0000 |
| Gemma 4 12B IT | A5 | 0.8180 | 0.8458 | 0.9498 | 0.2363 | 0.8948 | 0.5931 | 1.0000 |

> **Do not read Accuracy / F1 as skill for the collapsed rows.** InternVL3.5 has the highest Acc/F1 in the table only because 29–61% of samples are Uncertain (excluded) and almost none are Unreliable (Spec ≤ 0.144). Gemma 4 Accuracy (0.814–0.825) is within ±0.011 of the always-Reliable prior 0.8152. Prefer Specificity, BalAcc, Coverage and the R/U/Ur mix.

Experiment paths (`outputs/verification/…` and `outputs/evaluation/…` twins):

| Model | Experiment root |
|-------|-----------------|
| Qwen2.5-VL | `qwen/20260708_0020/` (legacy key) |
| Qwen3-VL | `qwen3_vl/qwen3vl_A1A5_5747/` |
| GLM-4.6V-Flash | `glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747/` |
| Phi-4 Multimodal | `phi4_multimodal/20260921_phi4_A1A5_5747/` |
| InternVL3.5-8B-HF | `internvl3_5_hf/20260924_internvl3_5_hf_A1A5_5747/` |
| Gemma 4 12B IT | `gemma4/20260925_gemma4_A1A5_5747/` |

---

## D. Prediction distributions

| Model | Cond | Reliable | Uncertain | Unreliable | TP / FP / FN / TN |
|-------|------|---------:|----------:|-----------:|-------------------|
| Qwen2.5-VL | A1 | 3593 | 1775 | 379 | 3214 / 379 / 212 / 167 |
| Qwen2.5-VL | A2 | 4268 | 1344 | 135 | 3754 / 514 / 75 / 60 |
| Qwen2.5-VL | A3 | 4416 | 1313 | 18 | 3866 / 550 / 8 / 10 |
| Qwen2.5-VL | A4 | 3570 | 1557 | 620 | 3237 / 333 / 366 / 254 |
| Qwen2.5-VL | A5 | 3065 | 455 | 2227 | 2791 / 274 / 1523 / 704 |
| Qwen3-VL | A1 | 4309 | 246 | 1192 | 3721 / 588 / 777 / 415 |
| Qwen3-VL | A2 | 4636 | 401 | 710 | 3966 / 670 / 442 / 268 |
| Qwen3-VL | A3 | 4221 | 367 | 1159 | 3651 / 570 / 758 / 401 |
| Qwen3-VL | A4 | 5051 | 276 | 420 | 4189 / 862 / 294 / 126 |
| Qwen3-VL | A5 | 1081 | 3948 | 718 | 1019 / 62 / 450 / 268 |
| GLM-4.6V-Flash | A1 | 3719 | 50 | 1978 | 3247 / 472 / 1403 / 575 |
| GLM-4.6V-Flash | A2 | 3783 | 57 | 1907 | 3319 / 464 / 1329 / 578 |
| GLM-4.6V-Flash | A3 | 3696 | 100 | 1951 | 3277 / 419 / 1345 / 606 |
| GLM-4.6V-Flash | A4 | 3960 | 64 | 1723 | 3502 / 458 / 1142 / 581 |
| GLM-4.6V-Flash | A5 | 2163 | 1082 | 2502 | 1990 / 173 / 1952 / 550 |
| Phi-4 Multimodal | A1 | 4985 | 0 | 762 | 4163 / 822 / 522 / 240 |
| Phi-4 Multimodal | A2 | 5012 | 0 | 735 | 4198 / 814 / 487 / 248 |
| Phi-4 Multimodal | A3 | 4132 | 0 | 1615 | 3534 / 598 / 1151 / 464 |
| Phi-4 Multimodal | A4 | 3481 | 0 | 2266 | 3134 / 347 / 1551 / 715 |
| Phi-4 Multimodal | A5 | 4160 | 0 | 1587 | 3658 / 502 / 1027 / 560 |
| InternVL3.5-8B-HF | A1 | 3085 | 2570 | 92 | 2775 / 310 / 40 / 52 |
| InternVL3.5-8B-HF | A2 | 3497 | 2239 | 11 | 3166 / 331 / 2 / 9 |
| InternVL3.5-8B-HF | A3 | 3420 | 2298 | 29 | 3104 / 316 / 7 / 22 |
| InternVL3.5-8B-HF | A4 | 3343 | 2399 | 5 | 3030 / 313 / 0 / 5 |
| InternVL3.5-8B-HF | A5 | 1515 | 4078 | 154 | 1398 / 117 / 137 / 17 |
| Gemma 4 12B IT | A1 | 5583 | 0 | 164 | 4604 / 979 / 81 / 83 |
| Gemma 4 12B IT | A2 | 5595 | 0 | 152 | 4605 / 990 / 80 / 72 |
| Gemma 4 12B IT | A3 | 5424 | 0 | 323 | 4538 / 886 / 147 / 176 |
| Gemma 4 12B IT | A4 | 5547 | 0 | 200 | 4614 / 933 / 71 / 129 |
| Gemma 4 12B IT | A5 | 5261 | 0 | 486 | 4450 / 811 / 235 / 251 |

---

## E. Scientific observations (descriptive)

1. **Qwen2.5-VL** remains non-collapsed across A1–A5 (uses all three labels). Adding confidence / geometry (A2→A3) raises recall and F1 but **collapses specificity** (A3 Spec = 0.0179). A5 is more conservative (highest Qwen2.5 Spec, lower recall/F1).

2. **Qwen3-VL** shows a strong condition-dependent shift. A1–A4 use a mixed R/U/Ur policy; **A5 becomes highly conservative**: R/U/Ur = 1081 / 3948 / 718, Precision 0.9426, Specificity 0.8121, but Coverage only 0.3130.

3. **GLM-4.6V-Flash** remains non-collapsed on every condition, with substantial Unreliable usage (~30–44%). Specificity stays in a mid–high band (0.549–0.761).

4. **Phi-4 Multimodal** is non-collapsed but **effectively binary** (zero Uncertain in every condition). Its A1 specificity is weak (0.2260; 0.1111 at A1@1000) but A4 reaches 0.6733 — A1 weakness alone did not predict failure.

5. **InternVL3.5-8B-HF** is an **abstention-heavy collapse** at full scale: Uncertain 39–71%, Unreliable ≤ 2.68%, Specificity 0.016–0.144, BalAcc ≤ 0.565 in every condition. A5 is the most abstaining (U = 4078). Its balanced-100 PASS did not predict full-scale behavior.

6. **Gemma 4 12B IT** is a **Reliable-heavy collapse** at full scale: Reliable 91.5–97.4%, zero Uncertain, Specificity 0.068–0.236, Accuracy ≈ prior. A5 is the least collapsed condition (Spec 0.2363). Gemma 4 did **not** follow the canonical qualification gate (canonical §6).

**Interpretation reminder:** always-Reliable Acc = 0.8152 on this set. Prefer specificity, balanced accuracy, coverage and the decision mix over Acc/F1 alone when judging second-stage FP rejection.

---

## F. A1 only (not peer A1–A5)

These runs are Complete at A1 @5747 but **A1 only** overall. A2–A5 were **Not evaluated**.

| Model | Experiment | R / U / Ur | Acc | Spec | BalAcc | Coverage | Outcome |
|-------|------------|------------|----:|-----:|-------:|---------:|---------|
| MiniCPM-V-4.5 | `20260923_minicpm_A1A5_5747` A1 (H200, job `8351080`) | 5557 / 0 / 190 | 0.8110 | 0.0782 | 0.5277 | 1.0000 | Reliable-heavy collapse (Acc ≈ prior) |
| Molmo2-8B | `20260923_molmo2_A1A5_5747` A1 (H200, job `8351081`) | 1209 / 4537 / 1 | 0.9264 | 0.0000 | 0.4996 | 0.2105 | Abstention-heavy collapse (F1 0.9618 on 1210 scored samples) |

**Earlier collapsed candidates (A1@1000 only):** LLaVA-OneVision and Gemma 3 → 1000/0/0 Reliable, Acc = 0.928 = slice prior, Spec = 0 (single-class Reliable-heavy collapse). Retained as negative controls.

**Qualification failed / stopped:** InternVL3-8B-Instruct (balanced-100: 13/100 parse failures, Spec 0).

---

## Related documents

| Doc | Role |
|-----|------|
| [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md) | Full inventory, jobs, qualification path, gate deviations, reproducibility |
| [`INTERNVL3_5_HF_QUALIFICATION.md`](INTERNVL3_5_HF_QUALIFICATION.md) | InternVL3 vs 3.5 balanced-100 gate + full-scale outcome |
| [`QWEN_FULL_A1_A5_RESULTS.md`](QWEN_FULL_A1_A5_RESULTS.md) | Qwen2.5-only detailed A1–A5 write-up |
| [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) | Matching and metric definitions |
| [`ABLATION_STUDY.md`](ABLATION_STUDY.md) | A1–A5 design |
