# Experiment Status — Canonical Run Inventory

**Date of original audit:** 2026-09-09 (DEAC cluster)  
**Last status update:** 2026-10-01 (construct correction: Protocol v2 reframed as LabelMe annotation alignment; no inference run, no run status changed)  
**Previous update:** 2026-09-27 (Evaluation Protocol v2 migration; no inference run)  
**Scope:** This file owns **run inventory, run status, scientific-outcome labels, reproducibility metadata and qualification history**. It deliberately contains no metric tables.

**Numeric results (single source of truth):** [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md)  
**Protocol:** [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) (current: **v2**) · [`ABLATION_STUDY.md`](ABLATION_STUDY.md) · [`FRAMEWORK_FREEZE.md`](FRAMEWORK_FREEZE.md)  
**Supporting analyses:** [`INTERNVL3_5_HF_QUALIFICATION.md`](INTERNVL3_5_HF_QUALIFICATION.md) · [`MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md`](MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md) · [`SEMANTIC_VALIDITY_AUDIT.md`](SEMANTIC_VALIDITY_AUDIT.md)

> [!IMPORTANT]
> **Construct correction — 2026-10-01 (interpretation only; no run status, verdict or number changed).**
>
> - **Protocol v2 is retained** as an **annotation-alignment reference**: 5,109 detections are LabelMe-matched (GT+), 638 are LabelMe-unmatched (GT−).
> - **LabelMe-unmatched does not mean human-confirmed non-palm.** The official human review of all 638 found **619 palm, 19 ambiguous, 0 non-palm** ([`SEMANTIC_VALIDITY_AUDIT.md`](SEMANTIC_VALIDITY_AUDIT.md)).
> - "Specificity" anywhere in this file means Protocol-v2 alignment specificity. Scientific-outcome labels (§0) are historical decision-mix descriptors under alignment scoring, not semantic-accuracy rankings.
> - **Qualification gates were alignment-based.** The 50 nominal "negative" examples of the Stage-1 balanced-100 set were later human-reviewed as **45 palm / 5 ambiguous / 0 non-palm**. Historical PASS/FAIL records in §5 are preserved as **alignment-gate outcomes**; they must not be read as semantic non-palm rejection tests (see §5, "Current scientific interpretation").

---

## 0. Vocabulary (two independent fields)

**Run status** — what was executed:

| Label | Meaning |
|-------|---------|
| **Complete** | A1–A5 @5747 all finished: 5747 prediction JSONs per condition, index status `ok`, evaluation CSV + metrics JSON present |
| **A1 only** | Only condition A1 was run at the stated scale; A2–A5 were not run |
| **Qualification failed / stopped** | Stopped at a qualification gate; no full-scale run |
| **Integrated; engineering qualification only** | Adapter/config/jobs exist and an engineering run validated the integration; no scientific evaluation |
| **Integrated; not evaluated** | Adapter/config/jobs exist; no experiment has been run |

**Scientific outcome** — what the run shows (descriptive; does not change the evaluator):

| Label | Operational meaning in this project |
|-------|-------------------------------------|
| **Useful verifier** | Uses rejection (and/or abstention) with substantial Protocol-v2 alignment specificity in at least one condition; not dominated by one class. Historical label; not a semantic-accuracy claim |
| **Abstention-heavy, condition-dependent** | Uncertain dominates and Unreliable is rare, but specificity on the decided subset is moderate in some conditions and near zero in others |
| **Reliable-heavy collapse** | Near-always-Reliable. *Single-class* = 100% Reliable (LLaVA, Gemma 3). *Partial* = Reliable ≥ ~91%, low specificity, accuracy ≈ always-Reliable prior |
| **Abstention-heavy collapse** | Uncertain dominates and Unreliable is essentially absent, so specificity on the scored subset is ≈ 0 |
| **Technical failure** | Parse/inference failures or a gate failure prevent a scientific read |

Poor Acc/F1 alone is **not** collapse. These labels summarize the observed decision mix; they are not a pass/fail gate and are not applied retroactively to qualification verdicts.

---

## 1. Protocol pointer

All results are scored with **Evaluation Protocol v2** (case-insensitive palm-label normalization; IoU ≥ 0.5; greedy one-to-one; Uncertain excluded). The v1 → v2 correction changed only GT labels, not predictions or R/U/Ur distributions, so **no run status changed**. Definitions: [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md); numbers and the correction record: [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md) §1–§2.

**Caveat for sub-5747 runs:** `reliable_pct` / `uncertain_pct` / `unreliable_pct` in `A*_metrics.json` are divided by 5747 even when fewer samples were evaluated. Use raw counts for @1000 / @100 runs.

---

## 2. Canonical checkpoint status table

| # | Model (checkpoint) | Run status | Scale / conditions | Scientific outcome | Registry key |
|---|--------------------|-----------|--------------------|--------------------|--------------|
| 1 | Qwen2.5-VL-7B-Instruct | **Complete** | A1–A5 @5747 | **Useful verifier** | `qwen2_5_vl` |
| 2 | Qwen3-VL-8B-Instruct | **Complete** | A1–A5 @5747 | **Useful verifier** | `qwen3_vl` |
| 3 | GLM-4.6V-Flash | **Complete** | A1–A5 @5747 | **Useful verifier** | `glm_4_6v_flash` |
| 4 | Phi-4-multimodal-instruct | **Complete** | A1–A5 @5747 | **Useful verifier** (binary — 0 Uncertain) | `phi4_multimodal` |
| 5 | InternVL3.5-8B-HF | **Complete** | A1–A5 @5747 | **Abstention-heavy; moderate specificity on A1 and A3 at low coverage, with near-zero specificity on A2, A4, and A5** | `internvl3_5_hf` |
| 6 | Gemma 4 12B IT | **Complete** | A1–A5 @5747 | **Reliable-heavy collapse** (partial) — gate deviation, see §5 | `gemma4` |
| 7 | MiniCPM-V-4.5 | **A1 only** | A1 @5747 | **Reliable-heavy collapse** (partial) | `minicpm_v4_5` |
| 8 | Molmo2-8B | **A1 only** | A1 @5747 | **Abstention-heavy collapse** | `molmo2_8b` |
| 9 | LLaVA-OneVision (Qwen2-7B OV) | **A1 only** | A1 @1000 | **Reliable-heavy collapse** (single-class) | `llava` |
| 10 | Gemma 3 12B IT | **A1 only** | A1 @1000 | **Reliable-heavy collapse** (single-class) | `gemma` |
| 11 | InternVL3-8B-Instruct | **Qualification failed / stopped** | Stage 0 + balanced-100 only | **Technical failure** (13/100 parse failures) | `internvl3` |
| 12 | Llama-3.2-11B-Vision-Instruct | **Integrated; engineering qualification only** | Engineering Stage 0 (10 samples) | — (no scientific result) | `llama3_2_11b_vision` |
| 13 | Ministral-3-8B | **Integrated; not evaluated** | — | — | `ministral3_8b` |

### Status matrix (full scale, @5747)

| Model | A1 | A2 | A3 | A4 | A5 |
|-------|----|----|----|----|----|
| Qwen2.5-VL-7B | Complete | Complete | Complete | Complete | Complete |
| Qwen3-VL-8B | Complete | Complete | Complete | Complete | Complete |
| GLM-4.6V-Flash | Complete | Complete | Complete | Complete | Complete |
| Phi-4 Multimodal | Complete | Complete | Complete | Complete | Complete |
| InternVL3.5-8B-HF | Complete | Complete | Complete | Complete | Complete |
| Gemma 4 12B IT | Complete | Complete | Complete | Complete | Complete |
| MiniCPM-V-4.5 | Complete | Not run | Not run | Not run | Not run |
| Molmo2-8B | Complete | Not run | Not run | Not run | Not run |
| InternVL3-8B, LLaVA-OV, Gemma 3, Llama-3.2-Vision, Ministral-3 | Not run @5747 | — | — | — | — |

Every Complete / A1-only cell was verified on 2026-09-27: exactly 5747 `sample_*.json`, `results_index.csv` all `ok`, **0 parse errors and 0 inference errors**, and metrics R/U/Ur equal to counts recomputed from the raw sample JSONs. The Protocol v2 re-score re-verified R/U/Ur and error counts for all 60 evaluated runs (`outputs/evaluation_protocol_v2/rescore_manifest.csv`).

---

## 3. Reproducibility inventory

| Model | Checkpoint (local → HF repo / revision if recorded) | Experiment ID (largest run) | Slurm jobs (A1→A5) | Hardware | Launch path |
|-------|------------------------------------------------------|-----------------------------|--------------------|----------|-------------|
| Qwen2.5-VL-7B | `…/models/Qwen2.5-VL-7B-Instruct` | `20260708_0020` (legacy tree `qwen/`) | historical ablation jobs; A5 resume `8318977` | yanggrp / L40S | `jobs/run_qwen_ablation.slurm` → `scripts/run_qwen_ablation_experiment.sh` |
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
| Llama-3.2-11B-Vision | `configs/models/llama3_2_11b_vision.yaml` | `20260927_llama3_2_11b_vision_engineering_qualification` (Stage 0, engineering) | `8353957` | A100 80GB | `jobs/run_llama3_2_11b_vision_engqual.slurm` (scientific gate: `jobs/run_llama3_2_11b_vision_qual.slurm`, not run) |
| Ministral-3-8B | `configs/models/ministral3_8b.yaml` | — | — | — | `jobs/download_ministral3_8b.slurm`, `jobs/run_ministral3_8b_qual.slurm` (not run) |

`…/models/` = `/deac/csc/yangGrp/luoz23/models/`. Paths (all gitignored, on-disk evidence only):

- Predictions: `outputs/verification/<registry_key>/<experiment_id>/<A1..A5>/`
- **Protocol v2 evaluation (current):** `outputs/evaluation_protocol_v2/<registry_key>/<experiment_id>/<A1..A5>/A*_metrics.json`
- Protocol v1 evaluation (frozen provenance): `outputs/evaluation/<same relative path>`

Runtime environments recorded in logs:

| Model | Python / env | transformers | torch |
|-------|--------------|--------------|-------|
| InternVL3.5-8B-HF | cluster `/usr/bin/python` 3.9.25 | 4.57.6 | (cluster default) |
| Gemma 4 12B IT | `/deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4` (Python 3.11.8; built by `jobs/setup_gemma4_env.slurm`) | 5.17.0 (requires ≥ 5.10.1, `AutoModelForMultimodalLM`) | 2.11.0+cu128 |
| GLM / Molmo2 / MiniCPM / Qwen3-VL | per-model venvs listed in `jobs/lib/model_runtime.sh::model_venv_path` | pinned in `jobs/run_verification.slurm` / `model_runtime.sh` | — |

Decoding for all adapters: greedy (`do_sample=False`), `max_new_tokens=512`. Gemma 4 runs with `enable_thinking=False`.

---

## 4. Reproducibility caveats (historical; outputs not altered)

- Gemma 4 sample JSONs have **no `generation` key** (the adapter did not emit generation metadata). Decoding settings must be read from `configs/models/gemma4.yaml` and `src/lvm/gemma4_verifier.py` (`do_sample=False`, `max_new_tokens=512`, `enable_thinking=False`). The same absence applies to historical Qwen2.5, Phi-4, LLaVA and Gemma 3 records. Do **not** back-fill this field.
- Gemma 4 records store `model_name` as the local checkpoint path; the upstream revision is recorded only in `…/models/gemma-4-12B-it/DOWNLOAD_META.txt` (table §3).
- Gemma 4's first A1@20 attempt `8353062` FAILED (`ModuleNotFoundError: pandas` in the new venv); fixed in `jobs/setup_gemma4_env.slurm` before `8353127`.
- InternVL3.5 and Molmo2 have high Acc/F1 because most samples are Uncertain and excluded; read them alongside coverage (see the results doc §5, §9).

---

## 5. Qualification path and historical gate deviations

The **preferred** gate for new models (documented after the InternVL3 audit) is: Stage 0 (~10 deliberate samples) → Stage 1 balanced-100 (50 GT+ / 50 GT− by Protocol-v2 alignment; parse ≥95%, not ≥95% one class, alignment Spec ≥0.20, alignment BalAcc ≥0.55) → A1@1000 → full A1–A5 @5747. **It was not applied uniformly.** Nothing below is retroactively changed; this section exists for transparency. All gate runs use samples from the first 1,000 detections and are unaffected by the Protocol v1 → v2 correction.

| Model | Actual path (evidence) | Balanced-100? | A1@1000? |
|-------|------------------------|:-------------:|:--------:|
| Qwen2.5-VL | Predates staged gate: A1–A5@1000 `20260706_2214` → A1–A5@5747 `20260708_0020` (A5 resumed, job `8318977`) | No | Yes (A1–A5) |
| Qwen3-VL | sanity1 → smoke20 → A1@1000 `qwen3vl_A1_1000` (`8349427`; qualified, no collapse: R/U/Ur = 724/48/228) → full | No | Yes |
| GLM-4.6V-Flash | Stage 0 (+ 0b / native) → balanced-100 `20260913_glm46v_flash_stage1_balanced100` → A1@1000 → A2–A5@1000 → full | **Yes** | Yes (A1–A5) |
| Phi-4 | gate0 load → smoke20 (`8340900`) → A1–A5@1000 (`8340904`–`8340908`; A2 re-run `8342097` after ECC-GPU TIMEOUT) → full | No | Yes (A1–A5) |
| InternVL3.5-8B-HF | Stage 0 (`8351986`, 10/10 ok) → balanced-100 PASS (`8351989`) → **directly** full A1–A5@5747 | **Yes** | **Skipped** |
| InternVL3-8B | Stage 0 → balanced-100 **FAIL_TECHNICAL** (`8303143`) → stopped | **Yes** | No |
| MiniCPM-V-4.5 | sanity1 → smoke20 → A1@1000 (`8350625`, L40S) → H200 smoke5 → A1@5747 (`8351080`) → stopped | No | Yes |
| Molmo2-8B | sanity1 → smoke20 → A1@1000 (`8350636`, L40S) → H200 smoke5 → A1@5747 (`8351081`) → stopped | No | Yes |
| LLaVA-OV | smoke / parity diagnostics → A1@1000 `20260719_1734` → stopped (collapse) | No | Yes |
| Gemma 3 | smoke → A1@1000 `20260802_1702` → stopped (collapse) | No | Yes |
| Gemma 4 | env setup (`8353043`) → A1@20 (`8353062` FAILED; `8353127` ok) → A1@100 resume (`8353129`) → **directly** full A1–A5@5747 | **No** | **Skipped** |
| Llama-3.2-11B-Vision | smoke + engineering Stage 0 (`8353957`, A100) → **stopped before the scientific gate** | No | No |
| Ministral-3-8B | integration only | No | No |

Documented deviations:

1. **InternVL3.5 skipped A1@1000.** It went from balanced-100 PASS straight to A1–A5@5747.
2. **Gemma 4 skipped Stage 0, balanced-100 and A1@1000.** Its only pre-check was `20260925_gemma4_qual` A1: the first 100 samples of the A1@1000 prompt index (92 GT+ / 8 GT−, **not** the balanced set), with R/U/Ur = **95 / 0 / 5**. 95% Reliable **meets the "≥95% one class" failure criterion** of the preferred Stage 1 gate, yet full scale was launched anyway. The full-scale run is a usable observation but did **not** follow the canonical sequence.
3. **Balanced-100 was only run for GLM, InternVL3 and InternVL3.5.** Qwen2.5, Qwen3-VL, Phi-4, MiniCPM, Molmo2, LLaVA, Gemma 3 and Gemma 4 never ran it.
4. **A1 weakness alone does not imply failure.** Phi-4's A1 specificity is below the Stage 1 floor at @1000, yet its A4 condition at full scale is among the strongest rejection profiles (results doc §5). Models stopped after A1 (MiniCPM, Molmo2, LLaVA, Gemma 3) were not tested on A2–A5.
5. **Llama-3.2-11B-Vision's Stage 0 run is engineering-only** (10 samples, 1 parse failure). It validates loading and parsing and is not a qualification verdict.

**Historical alignment-gate outcome vs current scientific interpretation (2026-10-01).**

| | Historical alignment-gate outcome (preserved) | Current scientific interpretation |
|---|---|---|
| What the gate measured | Parse rate, single-class share, and Spec / BalAcc against Protocol-v2 GT on 50 LabelMe-matched + 50 LabelMe-unmatched detections | Parse rate and single-class share are construct-neutral engineering/behavior checks. Spec / BalAcc measured agreement with LabelMe alignment only |
| The 50 "negatives" | Treated as GT− (assumed detector false positives) | Human-reviewed as **45 palm / 5 ambiguous / 0 non-palm**; not a non-palm rejection test |
| GLM-4.6V-Flash | PASS | Passed the alignment gate; says nothing about semantic non-palm rejection |
| InternVL3.5-8B-HF | PASS | Passed the alignment gate; says nothing about semantic non-palm rejection |
| InternVL3-8B | FAIL_TECHNICAL | Still a technical failure (13/100 unparsable), independent of the construct question |

No verdict is retroactively changed. Future gates should separate construct-neutral behavior checks from alignment-based checks and should not use LabelMe-unmatched detections as presumed non-palms.

---

## 6. Canonical datasets

| Asset | Path | Count |
|-------|------|------:|
| Production verification detections | `outputs/verification_dataset/` | **5747** |
| Shared ablation inputs | `outputs/verification_ablation_{10,100,1000,5747}/` | A1–A5 at each N |
| Balanced qualification gate | `outputs/diagnostics/model_qualification/balanced_A1_100/` | 50 LabelMe-matched / 50 LabelMe-unmatched (the 50 unmatched: 45 palm / 5 ambiguous / 0 non-palm by human review) |
| Official semantic audit (638 LabelMe-unmatched) | `outputs/semantic_gt_review/human_review.csv`; frozen copy `outputs/semantic_gt_review/archive/official_638_unmatched_review_completed_20261001T215529Z/` | 619 palm / 19 ambiguous / 0 non-palm |
| Lower-confidence semantic pilot (blind, stratified) | `outputs/semantic_gt_review/human_confidence_pilot.csv`; frozen copy `outputs/semantic_gt_review/archive/confidence_pilot_official_completed_20261001T172808Z/` | 400 (318 palm / 9 non-palm / 73 ambiguous) |
| Stage 0 deliberate set | `outputs/diagnostics/model_qualification/stage0_A1_10/` | 10 |
| Protocol v2 evaluation tree | `outputs/evaluation_protocol_v2/` | 60 runs (32 full-scale) |
| Protocol v1 evaluation tree (frozen) | `outputs/evaluation/` | 60 runs |

---

## 7. Open items

1. **Llama-3.2-11B-Vision** — run the scientific qualification gate (`jobs/run_llama3_2_11b_vision_qual.slurm`) before any scaling.
2. **Ministral-3-8B** — integrated; not evaluated.
3. **MiniCPM / Molmo2 A2–A5** — not run; not planned without a new scientific question.
4. **LLaVA / Gemma 3** — retained as negative controls only.
5. **Legacy per-model Slurm scripts** (`jobs/run_*_Ax_5747.slurm`, `jobs/run_qwen_ablation.slurm`, etc.) still default to the v1 path `outputs/evaluation/`; they carry a `DEPRECATED` header and are kept only to document how completed runs were launched. `jobs/run_verification.slurm` now defaults to v2 and refuses to write into `outputs/evaluation/`. For new runs use `scripts/submit_model_ablation.sh` → `jobs/run_verification.slurm`; re-score stored predictions with `scripts/rescore_protocol_v2.py`.
6. **Historical design docs** are archived under [`archive/docs/`](../archive/docs/README.md); their numbers are Protocol v1 and their plans are superseded.
7. **GT-independent A1–A5 behavior analysis** (primary analysis: R/U/Ur by model × condition, coverage/abstention, per-detection consistency, A1→A5 change rates, transition matrices, context sensitivity) — planned as a separate task from stored predictions only.
8. **Earlier semantic-GT pipeline** (`src/evaluation/semantic_gt.py`, `scripts/evaluate_semantic_gt.py`, `outputs/semantic_gt_evaluation/`) is **SUPERSEDED / NOT FOR CURRENT RESULTS**: it predates the official review workflow and never consumed its labels. See [`SEMANTIC_GT_EVALUATION.md`](SEMANTIC_GT_EVALUATION.md).

---

## End summary

<sub>Outcome labels below are historical decision-mix descriptors under Protocol-v2 alignment scoring (see the 2026-10-01 construct correction at the top).</sub>

**Complete (A1–A5 @5747), Useful verifier:** Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4.  
**Complete (A1–A5 @5747), abstention-heavy / condition-dependent:** InternVL3.5-8B-HF.  
**Complete (A1–A5 @5747), Reliable-heavy collapse:** Gemma 4 12B IT (gate deviation).  
**A1 only @5747, collapsed:** MiniCPM-V-4.5 (Reliable-heavy), Molmo2-8B (abstention-heavy).  
**A1 only @1000, single-class collapse:** LLaVA-OneVision, Gemma 3.  
**Qualification failed / stopped:** InternVL3-8B-Instruct (technical failure).  
**Integrated; engineering qualification only:** Llama-3.2-11B-Vision-Instruct.  
**Integrated; not evaluated:** Ministral-3-8B.
