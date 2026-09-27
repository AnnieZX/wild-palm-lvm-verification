# InternVL3.5-8B-HF Qualification Report

**Status:** Stage 0 + Stage 1 **PASS** → full A1–A5 @5747 **Complete** (run status) · **Abstention-heavy; moderate specificity on A1 and A3 at low coverage, with near-zero specificity on A2, A4, and A5** (scientific outcome, Evaluation Protocol v2)  
**Date:** 2026-09-24 (DEAC); full-scale outcome added 2026-09-27; re-adjudicated under Protocol v2 on 2026-09-27  
**Experiment ID:** `20260924_internvl3_5_hf_qual`  
**Related (failed prior stack):** [`INTERNVL3_QUALIFICATION.md`](INTERNVL3_QUALIFICATION.md)  
**Full-scale:** `20260924_internvl3_5_hf_A1A5_5747` (jobs `8351993`–`8351997`, all COMPLETED) — see §6  
**Checkpoint revision:** `741a7d03020411e666c6109218ab71e08151ef86` (from `DOWNLOAD_META.txt`)

---

## 1. Exact model

| Field | Value |
|-------|-------|
| **HF repo** | `OpenGVLab/InternVL3_5-8B-HF` |
| **Local path** | `/deac/csc/yangGrp/luoz23/models/InternVL3_5-8B-HF` |
| **Registry key** | `internvl3_5_hf` (aliases `internvl35_hf`, `internvl3.5_hf`) |
| **Config** | `configs/models/internvl3_5_hf.yaml` |
| **API** | HF-native: `AutoProcessor` + `AutoModelForImageTextToText` + `apply_chat_template` + `generate` |
| **trust_remote_code** | **False** (recorded in sample `generation` metadata) |
| **dtype** | bfloat16 |
| **Decoding** | `do_sample=False`, `max_new_tokens=512` |

---

## 2. Controlled contrast (same balanced-100 gate)

Both runs use `outputs/diagnostics/model_qualification/balanced_A1_100/`  
(seed `20260908`, 50 GT+ / 50 GT−).

| | InternVL3-8B-Instruct | InternVL3.5-8B-HF |
|--|----------------------:|------------------:|
| Experiment | `20260909_internvl3_qual` | `20260924_internvl3_5_hf_qual` |
| Stage 1 job | `8303143` (V100) | `8351989` (yanggrp / L40S / lovelace) |
| Implementation | remote-code `AutoModel` + `model.chat` | HF-native Transformers |
| N | 100 | 100 |
| Parse failures | **13** (87% parse success) | **0** (100% parse success) |
| Inference failures | 0 | 0 |
| R / U / Ur | **56 / 31 / 0** | **39 / 58 / 3** |
| TP / TN / FP / FN | 35 / 0 / 21 / 0 | **27 / 3 / 12 / 0** |
| Specificity | **0.00** | **0.20** |
| Balanced Accuracy | **0.50** | **0.60** |
| Sensitivity (binary) | 1.00 | 1.00 |
| Precision (binary) | 0.625 | 0.692 |
| Accuracy (binary) | 0.625 | 0.714 |
| F1 (binary) | 0.769 | 0.818 |
| Stage 1 verdict | **FAIL_TECHNICAL** | **PASS** |

**Source of truth:** reconstructed from `results_index.csv` + per-sample JSON under each `balanced100/` tree and the shared balanced manifest (not from secondary docs alone).

---

## 3. Stage 0

- Job **8351986** (yanggrp / L40S)
- Path: `outputs/verification/internvl3_5_hf/20260924_internvl3_5_hf_qual/stage0/`
- **10/10** `ok`; 0 parse / inference failures
- **PASS**

---

## 4. Stage 1 decision mix (observed)

| Decision | Count | % |
|----------|------:|--:|
| Reliable | 39 | 39% |
| Uncertain | 58 | 58% |
| Unreliable | 3 | 3% |

GT− destinations (n=50): Reliable 12 · Uncertain 35 · Unreliable 3.

---

## 5. Conservative interpretation (fact vs inference)

### Observed facts

1. Old InternVL3 Stage 1 had **13 JSON parse failures** (`Expecting property name enclosed in double quotes…` on failed samples) and **zero Unreliable**.
2. InternVL3.5-HF Stage 1 has **zero parse failures**, uses **all three labels**, and records **TN=3** (Spec=0.20).
3. InternVL3.5-HF is **Uncertain-heavy** (58/100).
4. Negative discrimination improved relative to InternVL3 Stage 1 but remains **weak** (Spec=0.20 meets the Stage 1 floor; BalAcc=0.60).

### Interpretations that are **not** proven by this contrast alone

- Improvement cannot be attributed solely to **model version** vs **HF-native implementation** — **both changed**.
- Uncertain-heavy behavior is **not** proven to be calibrated uncertainty; it is an observed abstention rate.
- That the Stage 1 PASS predicts full-scale behavior — §6 shows the full-scale profile is only partly consistent with it (A1/A3 specificity is moderate, A2/A4/A5 near zero).

---

## 6. Full-scale result (Complete)

Experiment `20260924_internvl3_5_hf_A1A5_5747` · jobs `8351993`–`8351997` all COMPLETED (yanggrp / L40S / lovelace; cluster Python 3.9.25, transformers 4.57.6) · 5747 records per condition · **0 parse / 0 inference errors**.

Metrics under Evaluation Protocol v2: [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md) §5 (row group "InternVL3.5-8B-HF") and §9.

Summary (Protocol v2):

- **Decision mix (protocol-independent):** Uncertain 38.96–70.96%, Unreliable 0.09–2.68%, coverage 0.29–0.61.
- **A1:** Spec 0.366, BalAcc 0.676 at coverage 0.55. **A3:** Spec 0.242, BalAcc 0.620 at coverage 0.60. These are moderate but rest on few decided GT− boxes (A1: 134; A3: 91).
- **A2, A4, A5:** Spec 0.106, 0.055, 0.094 — near zero.
- Acc 0.89–0.98 and F1 0.94–0.99 are computed on the non-Uncertain subset only and are **not** evidence of verification skill; the class prior is 0.889.

Protocol v1 (case-sensitive GT label) had reported Specificity 0.0157–0.1436 and classified the run as an abstention-heavy collapse in every condition. That classification is superseded; see [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md) §2.

**Qualification path deviation:** InternVL3.5 went Stage 0 → balanced-100 PASS → full A1–A5 @5747 and **skipped A1@1000**. See [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md) §5.

---

## 7. Validity verdict

| Run | Run status | Classification |
|-----|-----------|----------------|
| Stage 0 | Complete | **PASS** |
| Stage 1 balanced-100 | Complete | **PASS** (parse 100%; Spec≥0.20; BalAcc≥0.55; not ≥95% single class) |
| Full A1–A5 @5747 | **Complete** | **Abstention-heavy; moderate specificity on A1 and A3 at low coverage, with near-zero specificity on A2, A4, and A5** (valid full-scale observation; not classified as a Useful verifier) |
