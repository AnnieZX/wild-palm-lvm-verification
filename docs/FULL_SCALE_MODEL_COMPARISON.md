# Full-Scale Model Comparison (N = 5,747)

**Status date:** 2026-09-24  
**Source of truth:** verified on-disk prediction + evaluation trees (read-only audit).  
**Inventory / job IDs:** [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md)

This document is the **advisor-facing** summary of the frozen A1–A5 verification benchmark. It does **not** rank models or declare a winner.

**Status vocabulary:** **Complete** · **A1 only** · **Collapsed** · **Qualification failed / stopped** · **Not evaluated**.

---

## A. Experiment protocol

| Item | Definition |
|------|------------|
| **Task** | Second-stage VLM verification of fixed YOLO palm detections |
| **N** | **5,747** detections (full production set) |
| **GT** | LabelMe `palm` boxes; greedy 1–1 match, **IoU ≥ 0.5** |
| **GT prior** | GT+ = 4,685 · GT− = 1,062 · always-Reliable Acc ≈ **0.815** |
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

All cells in sections B–D for the four primary models are **Complete** and verified (5747 unique sample IDs, matching eval, 0 parse/inference errors).

---

## B. Main status table

| Model | A1 | A2 | A3 | A4 | A5 | Overall status |
|-------|----|----|----|----|----|----------------|
| **Qwen2.5-VL-7B** | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| **Qwen3-VL-8B** | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| **GLM-4.6V-Flash** | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| **Phi-4 Multimodal** | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| MiniCPM-V-4.5 | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated | **A1 only** (see §F) |
| Molmo2-8B | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated | **A1 only** (see §F) |

---

## C. Main metrics — four fully evaluated models

Specificity = TN / (TN + FP), derived from each condition’s evaluation counts.  
Uncertain predictions are **not** in these binary denominators.

| Model | Condition | Accuracy | Precision | Sensitivity | Specificity | F1 |
|-------|-----------|---------:|----------:|------------:|------------:|---:|
| Qwen2.5-VL | A1 | 0.8512 | 0.8945 | 0.9381 | 0.3059 | 0.9158 |
| Qwen2.5-VL | A2 | 0.8662 | 0.8796 | 0.9804 | 0.1045 | 0.9273 |
| Qwen2.5-VL | A3 | 0.8742 | 0.8755 | 0.9979 | 0.0179 | 0.9327 |
| Qwen2.5-VL | A4 | 0.8332 | 0.9067 | 0.8984 | 0.4327 | 0.9026 |
| Qwen2.5-VL | A5 | 0.6604 | 0.9106 | 0.6470 | 0.7198 | 0.7565 |
| Qwen3-VL | A1 | 0.7519 | 0.8635 | 0.8273 | 0.4138 | 0.8450 |
| Qwen3-VL | A2 | 0.7920 | 0.8555 | 0.8997 | 0.2857 | 0.8770 |
| Qwen3-VL | A3 | 0.7532 | 0.8650 | 0.8281 | 0.4130 | 0.8461 |
| Qwen3-VL | A4 | 0.7887 | 0.8293 | 0.9344 | 0.1275 | 0.8787 |
| Qwen3-VL | A5 | 0.7154 | 0.9426 | 0.6937 | 0.8121 | 0.7992 |
| GLM-4.6V-Flash | A1 | 0.6709 | 0.8731 | 0.6983 | 0.5492 | 0.7760 |
| GLM-4.6V-Flash | A2 | 0.6849 | 0.8773 | 0.7141 | 0.5547 | 0.7873 |
| GLM-4.6V-Flash | A3 | 0.6876 | 0.8866 | 0.7090 | 0.5912 | 0.7879 |
| GLM-4.6V-Flash | A4 | 0.7185 | 0.8843 | 0.7541 | 0.5592 | 0.8140 |
| GLM-4.6V-Flash | A5 | 0.5445 | 0.9200 | 0.5048 | 0.7607 | 0.6519 |
| Phi-4 Multimodal | A1 | 0.7661 | 0.8351 | 0.8886 | 0.2260 | 0.8610 |
| Phi-4 Multimodal | A2 | 0.7736 | 0.8376 | 0.8961 | 0.2335 | 0.8658 |
| Phi-4 Multimodal | A3 | 0.6957 | 0.8553 | 0.7543 | 0.4369 | 0.8016 |
| Phi-4 Multimodal | A4 | 0.6697 | 0.9003 | 0.6689 | 0.6733 | 0.7676 |
| Phi-4 Multimodal | A5 | 0.7339 | 0.8793 | 0.7808 | 0.5273 | 0.8271 |

Experiment paths:

| Model | Experiment ID |
|-------|---------------|
| Qwen2.5-VL | `outputs/verification/qwen/20260708_0020/` (+ eval twin) |
| Qwen3-VL | `…/qwen3_vl/qwen3vl_A1A5_5747/` |
| GLM-4.6V-Flash | `…/glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747/` |
| Phi-4 Multimodal | `…/phi4_multimodal/20260921_phi4_A1A5_5747/` |

---

## D. Prediction distributions

| Model | Condition | Reliable | Uncertain | Unreliable |
|-------|-----------|---------:|----------:|-----------:|
| Qwen2.5-VL | A1 | 3593 | 1775 | 379 |
| Qwen2.5-VL | A2 | 4268 | 1344 | 135 |
| Qwen2.5-VL | A3 | 4416 | 1313 | 18 |
| Qwen2.5-VL | A4 | 3570 | 1557 | 620 |
| Qwen2.5-VL | A5 | 3065 | 455 | 2227 |
| Qwen3-VL | A1 | 4309 | 246 | 1192 |
| Qwen3-VL | A2 | 4636 | 401 | 710 |
| Qwen3-VL | A3 | 4221 | 367 | 1159 |
| Qwen3-VL | A4 | 5051 | 276 | 420 |
| Qwen3-VL | A5 | 1081 | 3948 | 718 |
| GLM-4.6V-Flash | A1 | 3719 | 50 | 1978 |
| GLM-4.6V-Flash | A2 | 3783 | 57 | 1907 |
| GLM-4.6V-Flash | A3 | 3696 | 100 | 1951 |
| GLM-4.6V-Flash | A4 | 3960 | 64 | 1723 |
| GLM-4.6V-Flash | A5 | 2163 | 1082 | 2502 |
| Phi-4 Multimodal | A1 | 4985 | 0 | 762 |
| Phi-4 Multimodal | A2 | 5012 | 0 | 735 |
| Phi-4 Multimodal | A3 | 4132 | 0 | 1615 |
| Phi-4 Multimodal | A4 | 3481 | 0 | 2266 |
| Phi-4 Multimodal | A5 | 4160 | 0 | 1587 |

---

## E. Scientific observations (descriptive)

1. **Qwen2.5-VL** remains **non-collapsed** across A1–A5 (uses all three labels). Adding confidence / geometry (A2→A3) raises recall and F1 but **collapses specificity** (A3 Spec = 0.0179). A5 is more conservative (highest Spec among Qwen2.5 cells, lower recall/F1).

2. **Qwen3-VL** shows a strong **condition-dependent** shift. A1–A4 use a mixed R/U/Ur policy; **A5 becomes highly conservative**: R/U/Ur = **1081 / 3948 / 718**, Precision **0.9426**, Specificity **0.8121**, F1 **0.7992**. High Uncertain rate means binary metrics cover only the non-abstaining subset.

3. **GLM-4.6V-Flash** remains **non-collapsed** on every condition, with comparatively **substantial Unreliable** usage (~30–44% on A1–A5). Specificity stays in a mid–high band (~0.55–0.76) with a clear A5 precision/specificity vs recall tradeoff.

4. **Phi-4 Multimodal** is **non-collapsed** (non-trivial Unreliable counts; Spec up to 0.6733 on A4) but behaves as an **effectively binary** verifier: **zero Uncertain** predictions in every full-scale condition.

**Interpretation reminder:** Always-Reliable Acc ≈ 0.815 on this set. Prefer **specificity** (and decision mix) over Acc/F1 alone when judging second-stage FP rejection.

---

## F. A1 only (not peer A1–A5)

These runs are **Complete** at A1 @5747 but **A1 only** overall — not equivalent full-matrix comparisons. A2–A5 were **Not evaluated**.

| Model | Experiment | R / U / Ur | Acc | Spec | Note |
|-------|------------|------------|----:|-----:|------|
| **MiniCPM-V-4.5** | `20260923_minicpm_A1A5_5747` A1 (H200, job `8351080`) | 5557 / 0 / 190 | 0.8110 | **0.0782** | Near–always-Reliable (Acc ≈ class prior) |
| **Molmo2-8B** | `20260923_molmo2_A1A5_5747` A1 (H200, job `8351081`) | 1209 / 4537 / 1 | 0.9264 | **0.0000** | Uncertain-heavy; Spec 0 on binary subset |

**Earlier collapsed candidates (A1@1000 only):** LLaVA-OneVision and Gemma 3 → 100% Reliable, Spec = 0. Retained as negative-control evidence; see [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md).

---

## Related documents

| Doc | Role |
|-----|------|
| [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md) | Full inventory, jobs, taxonomy, history |
| [`QWEN_FULL_A1_A5_RESULTS.md`](QWEN_FULL_A1_A5_RESULTS.md) | Qwen2.5-only detailed A1–A5 write-up |
| [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) | Matching and metric definitions |
| [`ABLATION_STUDY.md`](ABLATION_STUDY.md) | A1–A5 design |
