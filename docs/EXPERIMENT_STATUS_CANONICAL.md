# Experiment Status — Canonical Source of Truth

**Date of original audit:** 2026-09-09 (DEAC cluster)  
**Last status update:** 2026-09-27 (repository canonicalization — every number below re-derived from raw `outputs/` trees and Slurm accounting; no inference, no output regeneration)  
**Scope:** This file is the single authoritative experiment inventory. Other docs summarize it; if they disagree, this file and the raw `outputs/evaluation/**/A*_metrics.json` + `outputs/verification/**/sample_*.json` win.

**Advisor-facing metric tables:** [`FULL_SCALE_MODEL_COMPARISON.md`](FULL_SCALE_MODEL_COMPARISON.md)  
**InternVL3 → InternVL3.5 gate:** [`INTERNVL3_5_HF_QUALIFICATION.md`](INTERNVL3_5_HF_QUALIFICATION.md)  
**Collapse forensics / model-selection history:** [`MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md`](MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md)  
**Protocol:** [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) · [`ABLATION_STUDY.md`](ABLATION_STUDY.md) · [`FRAMEWORK_FREEZE.md`](FRAMEWORK_FREEZE.md)

---

## 0. Vocabulary (two independent fields)

**Run status** — what was executed:

| Label | Meaning |
|-------|---------|
| **Complete** | A1–A5 @5747 all finished: 5747 prediction JSONs per condition, index status `ok`, evaluation CSV + metrics JSON present |
| **A1 only** | Only condition A1 was run at the stated scale; A2–A5 were not run |
| **Qualification failed / stopped** | Stopped at a qualification gate; no full-scale run |
| **Not evaluated** | No experiment run in this repository |

**Scientific outcome** — what the run shows (descriptive; does not change the frozen evaluator):

| Label | Operational meaning in this project |
|-------|-------------------------------------|
| **Useful verifier** | Uses rejection (and/or abstention) with non-trivial specificity in at least one condition; not dominated by one class |
| **Reliable-heavy collapse** | Near-always-Reliable. *Single-class* = ≥95% one class (LLaVA, Gemma 3 = 100%). *Partial* = multi-label but Reliable ≥ ~91%, Specificity near 0 and Accuracy ≈ always-Reliable prior |
| **Abstention-heavy collapse** | Uncertain dominates and Unreliable is nearly absent, so specificity on the scored subset is near 0; binary Acc/F1 are computed on a small scored subset and look inflated |
| **Technical failure** | Parse/inference failure rate or gate failure prevents a scientific read |

Poor Acc/F1 alone is **not** collapse. Collapse labels here are descriptive summaries of the observed decision mix; they are not a new pass/fail gate and are not applied retroactively to qualification verdicts.

---

## 1. Frozen protocol (unchanged since July 2026 freeze)

| Item | Definition | Implementation |
|------|------------|----------------|
| Dataset | **5,747** YOLO detections, YOLO confidence **≥ 0.5** | `outputs/verification_dataset/` |
| Ground truth | LabelMe `palm` boxes (evaluation only; never shown to the VLM) | `src/preprocessing/gt_palm_bboxes.py` |
| Matching | Pairwise IoU per image → sort descending → **greedy one-to-one** → match iff **IoU ≥ 0.5** | `src/evaluation/gt_matching.py`, `scripts/evaluate_verification_against_groundtruth.py` |
| GT prior | GT+ = **4,685** · GT− = **1,062** · always-Reliable Accuracy = 4685/5747 = **0.8152** | every `A*_metrics.json` (`ground_truth_positive/negative`) |
| Decisions | **Reliable** · **Uncertain** · **Unreliable** | shared parser `src/lvm/verification_response_parser.py` |
| Binary evaluation | Reliable = positive; Unreliable = negative; **Uncertain excluded** from Accuracy, Precision, Sensitivity, Specificity, F1 | `scripts/compute_verification_metrics.py` |
| Ablations | A1 overlay · A2 overlay + YOLO conf · A3 overlay + conf + geometry · A4 dual panel (overlay + crop) + conf · A5 crop only + conf | `src/prompts/ablation_verification_prompts.py`, `outputs/verification_ablation_{10,100,1000,5747}/` |

Confusion on the binary subset: TP = GT+ ∧ Reliable · FP = GT− ∧ Reliable · FN = GT+ ∧ Unreliable · TN = GT− ∧ Unreliable. Uncertain is neither FN nor TN.

Derived (documentation-only, **not** written by the frozen evaluator): Specificity = TN/(TN+FP); Balanced Accuracy = (Sens+Spec)/2; Decision Coverage = (R+Ur)/N; Abstention = U/N.

**Integrity check (2026-09-27):** `git status` / `git diff HEAD` show **no changes** to the parser, prompts, GT matching, evaluator scripts, runner/records, preprocessing, `FRAMEWORK_FREEZE.md`, or `EVALUATION_PROTOCOL.md`.

**Caveat for sub-5747 runs:** the stored `reliable_pct` / `uncertain_pct` / `unreliable_pct` fields in `A*_metrics.json` are divided by the 5747 dataset size even when fewer samples were evaluated (e.g. LLaVA A1@1000 shows `reliable_pct = 17.4`). Use the raw counts for @1000 / @100 runs. The evaluator is frozen and was not changed.

---

## 2. Canonical 11-checkpoint status table

| # | Model (checkpoint) | Run status | Scale / conditions | Scientific outcome | Registry key |
|---|--------------------|-----------|--------------------|--------------------|--------------|
| 1 | Qwen2.5-VL-7B-Instruct | **Complete** | A1–A5 @5747 | **Useful verifier** (PRIMARY) | `qwen2_5_vl` |
| 2 | Qwen3-VL-8B-Instruct | **Complete** | A1–A5 @5747 | **Useful verifier** (within-family support) | `qwen3_vl` |
| 3 | GLM-4.6V-Flash | **Complete** | A1–A5 @5747 | **Useful verifier** (PRIMARY) | `glm_4_6v_flash` |
| 4 | Phi-4-multimodal-instruct | **Complete** | A1–A5 @5747 | **Useful verifier** (PRIMARY; binary — 0 Uncertain) | `phi4_multimodal` |
| 5 | InternVL3.5-8B-HF | **Complete** | A1–A5 @5747 | **Abstention-heavy collapse** (partial) | `internvl3_5_hf` |
| 6 | InternVL3-8B-Instruct | **Qualification failed / stopped** | Stage 0 + balanced-100 only | **Technical failure** (13/100 parse failures; Spec 0) | `internvl3` |
| 7 | MiniCPM-V-4.5 | **A1 only** | A1 @5747 | **Reliable-heavy collapse** (partial) | `minicpm_v4_5` |
| 8 | Molmo2-8B | **A1 only** | A1 @5747 | **Abstention-heavy collapse** (partial) | `molmo2_8b` |
| 9 | LLaVA-OneVision (Qwen2-7B OV) | **A1 only** | A1 @1000 | **Reliable-heavy collapse** (single-class, 100% Reliable) | `llava` |
| 10 | Gemma 3 12B IT | **A1 only** | A1 @1000 | **Reliable-heavy collapse** (single-class, 100% Reliable) | `gemma` |
| 11 | Gemma 4 12B IT | **Complete** | A1–A5 @5747 | **Reliable-heavy collapse** (partial) — gate deviation, see §6 | `gemma4` |

**Next candidate / not yet evaluated:** Llama-3.2-11B-Vision-Instruct. No adapter, config, job script, download, or run exists in this repository.

### Status matrix (full scale, @5747)

| Model | A1 | A2 | A3 | A4 | A5 |
|-------|----|----|----|----|----|
| Qwen2.5-VL-7B | Complete | Complete | Complete | Complete | Complete |
| Qwen3-VL-8B | Complete | Complete | Complete | Complete | Complete |
| GLM-4.6V-Flash | Complete | Complete | Complete | Complete | Complete |
| Phi-4 Multimodal | Complete | Complete | Complete | Complete | Complete |
| InternVL3.5-8B-HF | Complete | Complete | Complete | Complete | Complete |
| Gemma 4 12B IT | Complete | Complete | Complete | Complete | Complete |
| MiniCPM-V-4.5 | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| Molmo2-8B | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| InternVL3-8B, LLaVA-OV, Gemma 3 | Not evaluated @5747 | — | — | — | — |

Every Complete / A1-only cell above was verified on 2026-09-27: exactly 5747 `sample_*.json`, `results_index.csv` all `ok`, **0 parse errors and 0 inference errors**, metrics JSON TP/FP/FN/TN and R/U/Ur equal to counts recomputed from the raw sample JSONs.

---

## 3. Reproducibility inventory

| Model | Checkpoint (local → HF repo / revision if recorded) | Experiment ID (largest run) | Slurm jobs (A1→A5) | Hardware | Launch path |
|-------|------------------------------------------------------|-----------------------------|--------------------|----------|-------------|
| Qwen2.5-VL-7B | `…/models/Qwen2.5-VL-7B-Instruct` | `20260708_0020` (legacy tree `outputs/verification/qwen/`) | historical ablation jobs; A5 resume `8318977` | yanggrp / L40S | `jobs/run_qwen_ablation.slurm` → `scripts/run_qwen_ablation_experiment.sh` |
| Qwen3-VL-8B | `…/models/Qwen3-VL-8B-Instruct` | `qwen3vl_A1A5_5747` | `8349703`–`8349707` | yanggrp / L40S | `scripts/submit_model_ablation.sh` → `jobs/run_verification.slurm` |
| GLM-4.6V-Flash | `…/models/GLM-4.6V-Flash` | `20260919_glm46v_flash_A1A5_5747` | `8340875`–`8340879` | yanggrp / L40S | `jobs/run_glm_4_6v_flash_Ax_5747.slurm` |
| Phi-4 Multimodal | `…/models/Phi-4-multimodal-instruct` | `20260921_phi4_A1A5_5747` | `8342381`–`8342385` | yanggrp / L40S | `jobs/submit_phi4_Ax_5747.sh` → `jobs/run_phi4_Ax_5747.slurm` |
| InternVL3.5-8B-HF | `…/models/InternVL3_5-8B-HF` → `OpenGVLab/InternVL3_5-8B-HF` @ `741a7d03020411e666c6109218ab71e08151ef86` | `20260924_internvl3_5_hf_A1A5_5747` | `8351993`–`8351997` | yanggrp / L40S (lovelace) | `jobs/submit_internvl3_5_hf_Ax_5747.sh` → `jobs/run_internvl3_5_hf_Ax_5747.slurm` |
| Gemma 4 12B IT | `…/models/gemma-4-12B-it` → `google/gemma-4-12B-it` @ `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | `20260925_gemma4_A1A5_5747` | `8353160`–`8353164` | yanggrp / L40S (lovelace) | `scripts/submit_model_ablation.sh gemma4_12b` → `jobs/run_verification.slurm` |
| MiniCPM-V-4.5 | `…/models/MiniCPM-V-4_5` (`openbmb/MiniCPM-V-4_5`) | `20260923_minicpm_A1A5_5747` (A1 only) | `8351080` | gpu_small / **H200** | `jobs/run_verification.slurm` |
| Molmo2-8B | `…/models/Molmo2-8B` (`allenai/Molmo2-8B`) | `20260923_molmo2_A1A5_5747` (A1 only) | `8351081` | gpu_small / **H200** | `jobs/run_verification.slurm` |
| InternVL3-8B | `…/models/InternVL3-8B-Instruct` | `20260909_internvl3_qual` | Stage 1 `8303143` | V100 | `jobs/run_internvl3_stage{0,1_balanced100}.slurm` |
| LLaVA-OneVision | `…/models/llava_onevision` | `20260719_1734` (A1@1000) | see `logs/slurm/llava_A1_20260719_1734.out` | yanggrp / L40S | `jobs/run_llava_A1.slurm` |
| Gemma 3 12B IT | `…/models/gemma-3-12b-it` | `20260802_1702` (A1@1000) | `8167800` (log `slurm-8167800.out`, repo root, gitignored) | yanggrp / L40S | `jobs/run_gemma_A1_1000.slurm` |

`…/models/` = `/deac/csc/yangGrp/luoz23/models/`. Paths: predictions `outputs/verification/<registry_key>/<experiment_id>/<A1..A5>/`, metrics `outputs/evaluation/<registry_key>/<experiment_id>/<A1..A5>/A*_metrics.json` (both gitignored; on-disk evidence only).

Runtime environments recorded in logs:

| Model | Python / env | transformers | torch |
|-------|--------------|--------------|-------|
| InternVL3.5-8B-HF | cluster `/usr/bin/python` 3.9.25 | 4.57.6 | (cluster default) |
| Gemma 4 12B IT | `/deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4` (Python 3.11.8; built by `jobs/setup_gemma4_env.slurm`) | 5.17.0 (requires ≥ 5.10.1, `AutoModelForMultimodalLM`) | 2.11.0+cu128 |
| GLM / Molmo2 / MiniCPM / Qwen3-VL | per-model venvs listed in `jobs/lib/model_runtime.sh::model_venv_path` | pinned in `jobs/run_verification.slurm` / `model_runtime.sh` | — |

Decoding for all adapters: greedy (`do_sample=False`), `max_new_tokens=512`. Gemma 4 runs with `enable_thinking=False`.

---

## 4. InternVL3.5-8B-HF — full A1–A5 @5747 (Complete)

**Run status:** Complete · **Scientific outcome:** Abstention-heavy collapse (partial)  
Experiment `20260924_internvl3_5_hf_A1A5_5747` · jobs `8351993`–`8351997` all COMPLETED (2026-09-24/25) · 5747 records per condition · **0 parse / 0 inference errors**.

| Cond | R / U / Ur | R% / U% / Ur% | TP / FP / FN / TN | Acc | Prec | Sens | Spec | F1 | BalAcc | Coverage |
|------|-----------|---------------|-------------------|----:|-----:|-----:|-----:|---:|-------:|---------:|
| A1 | 3085 / 2570 / 92 | 53.68 / 44.72 / 1.60 | 2775 / 310 / 40 / 52 | 0.8898 | 0.8995 | 0.9858 | 0.1436 | 0.9407 | 0.5647 | 0.5528 |
| A2 | 3497 / 2239 / 11 | 60.85 / 38.96 / 0.19 | 3166 / 331 / 2 / 9 | 0.9051 | 0.9053 | 0.9994 | 0.0265 | 0.9500 | 0.5129 | 0.6104 |
| A3 | 3420 / 2298 / 29 | 59.51 / 39.99 / 0.50 | 3104 / 316 / 7 / 22 | 0.9063 | 0.9076 | 0.9977 | 0.0651 | 0.9505 | 0.5314 | 0.6001 |
| A4 | 3343 / 2399 / 5 | 58.17 / 41.74 / 0.09 | 3030 / 313 / 0 / 5 | 0.9065 | 0.9064 | 1.0000 | 0.0157 | 0.9509 | 0.5079 | 0.5826 |
| A5 | 1515 / 4078 / 154 | 26.36 / 70.96 / 2.68 | 1398 / 117 / 137 / 17 | 0.8478 | 0.9228 | 0.9107 | 0.1269 | 0.9167 | 0.5188 | 0.2904 |

Ranges: Uncertain **38.96–70.96%**, Unreliable **0.09–2.68%**, Specificity **0.0157–0.1436**, Balanced Accuracy **0.508–0.565**.

**Warning:** Acc 0.85–0.91 and F1 0.92–0.95 are **higher** than every Useful-verifier cell but are computed only on the 29–61% of samples that were not Uncertain, and almost none of those are Unreliable. They do **not** indicate verification skill. The Stage 1 balanced-100 PASS (Spec 0.20, BalAcc 0.60, U = 58/100) did not carry over to a non-collapsed full-scale profile.

---

## 5. Gemma 4 12B IT — full A1–A5 @5747 (Complete)

**Run status:** Complete · **Scientific outcome:** Reliable-heavy collapse (partial)  
Experiment `20260925_gemma4_A1A5_5747` · jobs `8353160`–`8353164` all COMPLETED (2026-09-25/26) · 5747 records per condition · **0 parse / 0 inference errors** · registry key `gemma4` (submitted via alias `gemma4_12b`).

| Cond | R / U / Ur | R% / U% / Ur% | TP / FP / FN / TN | Acc | Prec | Sens | Spec | F1 | BalAcc | Coverage |
|------|-----------|---------------|-------------------|----:|-----:|-----:|-----:|---:|-------:|---------:|
| A1 | 5583 / 0 / 164 | 97.15 / 0.00 / 2.85 | 4604 / 979 / 81 / 83 | 0.8156 | 0.8246 | 0.9827 | 0.0782 | 0.8968 | 0.5304 | 1.0000 |
| A2 | 5595 / 0 / 152 | 97.36 / 0.00 / 2.64 | 4605 / 990 / 80 / 72 | 0.8138 | 0.8231 | 0.9829 | 0.0678 | 0.8959 | 0.5254 | 1.0000 |
| A3 | 5424 / 0 / 323 | 94.38 / 0.00 / 5.62 | 4538 / 886 / 147 / 176 | 0.8203 | 0.8367 | 0.9686 | 0.1657 | 0.8978 | 0.5672 | 1.0000 |
| A4 | 5547 / 0 / 200 | 96.52 / 0.00 / 3.48 | 4614 / 933 / 71 / 129 | 0.8253 | 0.8318 | 0.9848 | 0.1215 | 0.9019 | 0.5532 | 1.0000 |
| A5 | 5261 / 0 / 486 | 91.54 / 0.00 / 8.46 | 4450 / 811 / 235 / 251 | 0.8180 | 0.8458 | 0.9498 | 0.2363 | 0.8948 | 0.5931 | 1.0000 |

Ranges: Reliable **91.54–97.36%**, Uncertain **0** in every condition, Specificity **0.0678–0.2363**, Accuracy **0.8138–0.8253** vs always-Reliable prior **0.8152**.

**Warning:** Accuracy is within ±0.011 of the always-Reliable prior in every condition and F1 ≈ 0.90 is what a near-always-Reliable policy produces on this 81.5%-positive set. The run is a valid full-scale **observation** of Google-lineage behavior, not evidence of verification skill. Gemma 4 A1 is nearly indistinguishable from MiniCPM A1 (both FP = 979, TN = 83).

**Reproducibility caveats (historical; outputs not altered):**

- Gemma 4 sample JSONs have **no `generation` key** (the adapter did not emit generation metadata). Decoding settings must be read from `configs/models/gemma4.yaml` and `src/lvm/gemma4_verifier.py` (`do_sample=False`, `max_new_tokens=512`, `enable_thinking=False`). The same absence applies to historical Qwen2.5, Phi-4, LLaVA and Gemma 3 records. Do **not** back-fill this field.
- Records store `model_name` as the local checkpoint path; the upstream revision is recorded only in `…/models/gemma-4-12B-it/DOWNLOAD_META.txt` (table §3).
- First A1@20 attempt `8353062` FAILED (`ModuleNotFoundError: pandas` in the new venv); fixed in `jobs/setup_gemma4_env.slurm` before `8353127`.

---

## 6. Qualification path and historical gate deviations

The **preferred** gate for new models (documented after the InternVL3 audit) is: Stage 0 (~10 deliberate samples) → Stage 1 balanced-100 (50 GT+ / 50 GT−; parse ≥95%, not ≥95% one class, Spec ≥0.20, BalAcc ≥0.55) → A1@1000 → full A1–A5 @5747. **It was not applied uniformly.** Nothing below is retroactively changed; this section exists for transparency.

| Model | Actual path (evidence) | Balanced-100? | A1@1000? |
|-------|------------------------|:-------------:|:--------:|
| Qwen2.5-VL | Predates staged gate: A1–A5@1000 `20260706_2214` → A1–A5@5747 `20260708_0020` (A5 resumed, job `8318977`) | No | Yes (A1–A5) |
| Qwen3-VL | sanity1 → smoke20 → A1@1000 `qwen3vl_A1_1000` (`8349427`; qualified, no collapse: R/U/Ur = 724/48/228, Spec = 0.5077) → full | No | Yes |
| GLM-4.6V-Flash | Stage 0 (+ 0b / native) → balanced-100 `20260913_glm46v_flash_stage1_balanced100` → A1@1000 → A2–A5@1000 → full | **Yes** | Yes (A1–A5) |
| Phi-4 | gate0 load → smoke20 (`8340900`) → A1–A5@1000 (`8340904`–`8340908`; A2 re-run `8342097` after ECC-GPU TIMEOUT) → full | No | Yes (A1–A5) |
| InternVL3.5-8B-HF | Stage 0 (`8351986`, 10/10 ok) → balanced-100 PASS (`8351989`) → **directly** full A1–A5@5747 | **Yes** | **Skipped** |
| InternVL3-8B | Stage 0 → balanced-100 **FAIL_TECHNICAL** (`8303143`) → stopped | **Yes** | No |
| MiniCPM-V-4.5 | sanity1 → smoke20 → A1@1000 (`8350625`, L40S) → H200 smoke5 → A1@5747 (`8351080`) → stopped | No | Yes |
| Molmo2-8B | sanity1 → smoke20 → A1@1000 (`8350636`, L40S) → H200 smoke5 → A1@5747 (`8351081`) → stopped | No | Yes |
| LLaVA-OV | smoke / parity diagnostics → A1@1000 `20260719_1734` → stopped (collapse) | No | Yes |
| Gemma 3 | smoke → A1@1000 `20260802_1702` → stopped (collapse) | No | Yes |
| Gemma 4 | env setup (`8353043`) → A1@20 (`8353062` FAILED; `8353127` ok) → A1@100 resume (`8353129`) → **directly** full A1–A5@5747 | **No** | **Skipped** |

Documented deviations:

1. **InternVL3.5 skipped A1@1000.** It went from balanced-100 PASS straight to A1–A5@5747.
2. **Gemma 4 skipped Stage 0, balanced-100 and A1@1000.** Its only pre-check was `20260925_gemma4_qual` A1: the first 100 samples of the A1@1000 prompt index (92 GT+ / 8 GT−, **not** the balanced set). Result: R/U/Ur = **95 / 0 / 5**, TP/FP/FN/TN = 90/5/2/3, Spec = 0.375 on only 8 negatives. 95% Reliable **meets the "≥95% one class" failure criterion** of the preferred Stage 1 gate, yet full scale was launched anyway. The full-scale run is therefore a usable observation, but it did **not** follow the canonical qualification sequence and the Reliable-heavy pre-check was not treated as a stop signal.
3. **Balanced-100 was only run for GLM, InternVL3 and InternVL3.5.** Qwen2.5, Qwen3-VL, Phi-4, MiniCPM, Molmo2, LLaVA, Gemma 3 and Gemma 4 never ran it.
4. **A1 weakness alone does not imply failure.** Phi-4 A1@1000 Spec = **0.1111** (below the Stage 1 Spec ≥0.20 floor) and A1@5747 Spec = **0.2260**, but A4@5747 reaches Spec = **0.6733** / BalAcc = **0.6711**. Models stopped after A1 (MiniCPM, Molmo2, LLaVA, Gemma 3) were not tested on A2–A5.

---

## 7. Established models (unchanged; re-verified 2026-09-27)

A1 summary at the largest run (full A1–A5 tables in [`FULL_SCALE_MODEL_COMPARISON.md`](FULL_SCALE_MODEL_COMPARISON.md)):

| Model | Scale | A1 R / U / Ur | Acc | Prec | Sens | Spec | F1 | BalAcc | Coverage |
|-------|-------|---------------|----:|-----:|-----:|-----:|---:|-------:|---------:|
| Qwen2.5-VL-7B | @5747 | 3593 / 1775 / 379 | 0.8512 | 0.8945 | 0.9381 | 0.3059 | 0.9158 | 0.6220 | 0.6911 |
| Qwen3-VL-8B | @5747 | 4309 / 246 / 1192 | 0.7519 | 0.8635 | 0.8273 | 0.4138 | 0.8450 | 0.6205 | 0.9572 |
| GLM-4.6V-Flash | @5747 | 3719 / 50 / 1978 | 0.6709 | 0.8731 | 0.6983 | 0.5492 | 0.7760 | 0.6237 | 0.9913 |
| Phi-4 Multimodal | @5747 | 4985 / 0 / 762 | 0.7661 | 0.8351 | 0.8886 | 0.2260 | 0.8610 | 0.5573 | 1.0000 |
| MiniCPM-V-4.5 | @5747 | 5557 / 0 / 190 | 0.8110 | 0.8238 | 0.9772 | 0.0782 | 0.8940 | 0.5277 | 1.0000 |
| Molmo2-8B | @5747 | 1209 / 4537 / 1 | 0.9264 | 0.9272 | 0.9991 | 0.0000 | 0.9618 | 0.4996 | 0.2105 |
| LLaVA-OneVision | @1000 | 1000 / 0 / 0 | 0.9280 | 0.9280 | 1.0000 | 0.0000 | 0.9627 | 0.5000 | 1.0000 |
| Gemma 3 12B IT | @1000 | 1000 / 0 / 0 | 0.9280 | 0.9280 | 1.0000 | 0.0000 | 0.9627 | 0.5000 | 1.0000 |

TP/FP/FN/TN (A1): Qwen2.5 3214/379/212/167 · Qwen3 3721/588/777/415 · GLM 3247/472/1403/575 · Phi-4 4163/822/522/240 · MiniCPM 4578/979/107/83 · Molmo2 1121/88/1/0 · LLaVA 928/72/0/0 · Gemma 3 928/72/0/0.

InternVL3-8B (balanced-100, `20260909_internvl3_qual`): R/U/Ur 56/31/0 + **13 parse failures**; TP/TN/FP/FN 35/0/21/0; Spec 0.00; BalAcc 0.50 → **FAIL_TECHNICAL**. InternVL3.5 on the same gate: 39/58/3, 0 parse failures, Spec 0.20, BalAcc 0.60 → PASS.

Always-Reliable prior: @5747 Acc = 0.8152; @1000 A1 slice Acc = 0.928 (928/1000). Collapsed models' accuracy equals these priors.

---

## 8. Canonical datasets

| Asset | Path | Count |
|-------|------|------:|
| Production verification detections | `outputs/verification_dataset/` | **5747** |
| Shared ablation inputs | `outputs/verification_ablation_{10,100,1000,5747}/` | A1–A5 at each N |
| Balanced qualification gate | `outputs/diagnostics/model_qualification/balanced_A1_100/` | 50 / 50 |
| Stage 0 deliberate set | `outputs/diagnostics/model_qualification/stage0_A1_10/` | 10 |

---

## 9. Open items

1. **Next independent-family candidate:** Llama-3.2-11B-Vision-Instruct — *next candidate / not yet evaluated*. Not integrated; no adapter, config, or job exists.
2. **MiniCPM / Molmo2 A2–A5** — Not evaluated; not planned without a new scientific question.
3. **LLaVA / Gemma 3** — retained as negative controls only.
4. **Historical design docs** (`MULTI_MODEL_INTEGRATION_PLAN.md`, `VLM_MODEL_SELECTION_SURVEY.md`, `GEMMA_*`, `ADVISOR_RESEARCH_STATUS_20260913.md`, `GLM_A1_A5_1000_RESULTS.md`, `QWEN_FULL_A1_A5_RESULTS.md`) are dated snapshots; their "planned"/"pending" language is historical and superseded by this file.

---

## End summary

**Complete (A1–A5 @5747), Useful verifier:** Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4.  
**Complete (A1–A5 @5747), collapsed:** InternVL3.5-8B-HF (abstention-heavy), Gemma 4 12B IT (Reliable-heavy; gate deviation).  
**A1 only @5747, collapsed:** MiniCPM-V-4.5 (Reliable-heavy), Molmo2-8B (abstention-heavy).  
**A1 only @1000, single-class collapse:** LLaVA-OneVision, Gemma 3.  
**Qualification failed / stopped:** InternVL3-8B-Instruct (technical failure).  
**Not evaluated:** Llama-3.2-11B-Vision-Instruct (next candidate).
