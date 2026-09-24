# Experiment Status — Canonical Source of Truth

**Date of original audit:** 2026-09-09 (DEAC cluster)  
**Last status update:** 2026-09-24 (read-only full-scale verification audit)  
**Scope:** Documentation reflects **verified on-disk** prediction + evaluation trees. No new inference in this update.

**Advisor-facing metrics summary:** [`docs/FULL_SCALE_MODEL_COMPARISON.md`](FULL_SCALE_MODEL_COMPARISON.md)  
**Audit artifacts (A1@1000 triad):** `outputs/analysis/a1_1000_cross_model_audit/`

**Status vocabulary:** **Complete** · **A1 only** · **Collapsed** · **Qualification failed / stopped** · **Not evaluated**.

---

## 0. Current state (2026-09-24)

**Slurm:** no jobs RUNNING or PENDING for this project as of the 2026-09-24 audit.

### Full A1–A5 @5747 — Complete (verified)

Each cell below has **exactly 5747** prediction JSONs, **5747** unique `sample_id`s (contiguous `sample_000001`…`sample_005747`), matching evaluation CSV, and metrics JSON; index status `ok`; **0** parse / inference errors.

| Model | Checkpoint | Experiment ID | Verification root | Evaluation root | Slurm (A1→A5) | Partition / GPU |
|-------|------------|---------------|-------------------|-----------------|---------------|-----------------|
| **Qwen2.5-VL-7B-Instruct** | `/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct` | `20260708_0020` | `outputs/verification/qwen/20260708_0020/` | `outputs/evaluation/qwen/20260708_0020/` | Historical ablation + A5 resume `8318977` COMPLETED | yangGrp / L40S |
| **Qwen3-VL-8B-Instruct** | `…/models/Qwen3-VL-8B-Instruct` | `qwen3vl_A1A5_5747` | `outputs/verification/qwen3_vl/qwen3vl_A1A5_5747/` | `outputs/evaluation/qwen3_vl/qwen3vl_A1A5_5747/` | `8349703`–`8349707` COMPLETED | yangGrp / L40S |
| **GLM-4.6V-Flash** | `…/models/GLM-4.6V-Flash` | `20260919_glm46v_flash_A1A5_5747` | `outputs/verification/glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747/` | `outputs/evaluation/glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747/` | `8340875`–`8340879` COMPLETED | yangGrp / L40S |
| **Phi-4-multimodal-instruct** | `…/models/Phi-4-multimodal-instruct` | `20260921_phi4_A1A5_5747` | `outputs/verification/phi4_multimodal/20260921_phi4_A1A5_5747/` | `outputs/evaluation/phi4_multimodal/20260921_phi4_A1A5_5747/` | `8342381`–`8342385` COMPLETED | yangGrp / L40S |

**Explicit corrections vs older docs:**

| Historical / superseded claim (do not treat as current) | Verified reality |
|--------------------------------------------------------|------------------|
| Qwen2.5 A5 @5747 incomplete (TIME LIMIT) | **Complete** — 5747/5747 + eval + metrics under `20260708_0020/A5/` |
| Phi-4 A1–A5 @5747 not submitted | **Complete** — experiment `20260921_phi4_A1A5_5747` |
| Qwen3-VL full scale not started | **Complete** — experiment `qwen3vl_A1A5_5747` |
| MiniCPM / Molmo “no inference” | **A1 only** — A1 @5747 Complete on H200 (see below) |

### A1 @5747 only — Complete (verified); A2–A5 Not evaluated

| Model | Checkpoint | Experiment ID | Path (A1) | Job | Partition / GPU | Behavior note |
|-------|------------|---------------|-----------|-----|-----------------|---------------|
| **MiniCPM-V-4.5** | `…/models/MiniCPM-V-4_5` | `20260923_minicpm_A1A5_5747` | `…/minicpm_v4_5/20260923_minicpm_A1A5_5747/A1/` | `8351080` COMPLETED | gpu_small / **H200** | Near–always-Reliable (R=5557, Spec=0.0782) |
| **Molmo2-8B** | `…/models/Molmo2-8B` | `20260923_molmo2_A1A5_5747` | `…/molmo2_8b/20260923_molmo2_A1A5_5747/A1/` | `8351081` COMPLETED | gpu_small / **H200** | Uncertain-heavy (U=4537, Spec=0.0000) |

L40S submits `8351024` / `8351025` were **CANCELLED** before start; superseded by the H200 jobs above. Do **not** treat MiniCPM/Molmo as A1–A5 peers.

### Status matrix (full scale)

| Model | A1 | A2 | A3 | A4 | A5 | Overall |
|-------|----|----|----|----|----|---------|
| Qwen2.5-VL-7B | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| Qwen3-VL-8B | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| GLM-4.6V-Flash | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| Phi-4 Multimodal | Complete | Complete | Complete | Complete | Complete | **Complete** (A1–A5 @5747) |
| MiniCPM-V-4.5 | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated | **A1 only** |
| Molmo2-8B | Complete | Not evaluated | Not evaluated | Not evaluated | Not evaluated | **A1 only** |

---

## 1. Canonical datasets

| Asset | Path | Count | Status |
|-------|------|------:|--------|
| Production verification detections | `outputs/verification_dataset/` | **5747** images + prompts + metadata | COMPLETE |
| Index | `outputs/verification_dataset/index.csv` | 5747 | COMPLETE |
| Shared ablation inputs | `outputs/verification_ablation_{10,100,1000,5747}/` | A1–A5 prompts complete at each N | COMPLETE |

A1 overlay images resolve via `prompt_index.csv` → `../../verification_dataset/images/…` (0 missing for N=1000).

**Full-set GT prior:** GT+ = 4685 · GT− = 1062 · always-Reliable accuracy ≈ **0.815**.

---

## 2. Experiment taxonomy (keep history unambiguous)

| Category | Status term | What it is | Examples |
|----------|-------------|------------|----------|
| **Canonical primary** | **Complete** | Verified full A1–A5 @5747 | Qwen2.5, Qwen3-VL, GLM, Phi-4 |
| **Partial full-scale** | **A1 only** | Verified A1@5747 only; A2–A5 Not evaluated | MiniCPM, Molmo2 |
| **Qualification / subset** | (subset Complete) | Smoke, A1@1000, balanced gates | Qwen3 A1@1000; Phi-4 @1000; GLM@1000 |
| **Collapsed candidates** | **Collapsed** | Inference OK but scientifically degenerate | LLaVA A1@1000; Gemma A1@1000 |
| **Failed technical qualification** | **Qualification failed / stopped** | Did not pass Stage 1 | InternVL3 balanced-100 |
| **Superseded runtime** | historical | Hardware/timeout; later replaced | Phi-4 A2@1000 V100 TIMEOUT; MiniCPM/Molmo L40S CANCELLED |

---

## 3. Qualification and collapse evidence (retained)

### A1 @1000 triad (same `sample_id` set and order)

Confirmed: `results_index.csv` IDs for Qwen, LLaVA, and Gemma match  
`outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv`  
(`sample_000001` … `sample_001000`).

| Model | Experiment ID | N | Mix (R / U / Ur) | Spec | Verdict |
|-------|---------------|--:|------------------|-----:|---------|
| **Qwen2.5-VL** | `20260706_2214` | 1000 | 672 / 272 / 56 | 0.24 | Non-collapse baseline |
| **LLaVA-OneVision** | `20260719_1734` | 1000 | **1000 / 0 / 0** | **0** | **Collapsed** (= always-Reliable Acc 0.928) |
| **Gemma 3 12B IT** | `20260802_1702` | 1000 | **1000 / 0 / 0** | **0** | **Collapsed** (= always-Reliable Acc 0.928) |

GT on this slice: GT+ = 928 (92.8%), GT− = 72. High accuracy alone is not verification skill.

### Other qualification highlights

| Model | Experiment | Result |
|-------|------------|--------|
| **Qwen3-VL** A1@1000 | `qwen3vl_A1_1000` | Qualified (no collapse): R/U/Ur = 724/48/228; Spec 0.5077 |
| **Phi-4** A1/A3/A4/A5 @1000 | `20260919_1524_phi4_A1A5_1000` | COMPLETE |
| **Phi-4** A2 @1000 (original) | same exp `A2/` | **TIMEOUT** on ECC-bad V100 (`8340905`); 5/1000 |
| **Phi-4** A2 @1000 (clean) | `20260920_2339_phi4_A2_1000` | COMPLETE on L40S (`8342097`) |
| **GLM** A1–A5 @1000 | `20260913_…` / `20260914_…` | COMPLETE (pre–full-scale) |
| **InternVL3** Stage 0 | `20260909_internvl3_qual/stage0` | Technical PASS (10/10) |
| **InternVL3** Stage 1 | `…/balanced100` | **FAIL_TECHNICAL** (13% parse fail; Spec=0; not A1-1000-qualified) |
| **MiniCPM / Molmo** | sanity1 → smoke20 → A1@1000 | Completed before A1@5747 |

Full InternVL write-up: [`docs/INTERNVL3_QUALIFICATION.md`](INTERNVL3_QUALIFICATION.md).

---

## 4. Validity verdicts (updated)

| Run | Classification | Evidence |
|-----|----------------|----------|
| Qwen2.5 A1–A5 @1000 (`20260706_2214`) | **VALID_PRIMARY_RESULT** | Full ablation completed + evaluated |
| Qwen2.5 A1–A5 @5747 (`20260708_0020`) | **VALID_PRIMARY_RESULT** | All five conditions **5747/5747** verified (A5 **complete**) |
| Qwen3-VL A1–A5 @5747 (`qwen3vl_A1A5_5747`) | **VALID_PRIMARY_RESULT** | All five conditions verified |
| GLM A1–A5 @5747 (`20260919_glm46v_flash_A1A5_5747`) | **VALID_PRIMARY_RESULT** | All five conditions verified |
| Phi-4 A1–A5 @5747 (`20260921_phi4_A1A5_5747`) | **VALID_PRIMARY_RESULT** | All five conditions verified; 0 Uncertain |
| MiniCPM A1 @5747 | **VALID_BUT_LIMITED** | Complete run; near–always-Reliable → weak verifier evidence |
| Molmo2 A1 @5747 | **VALID_BUT_LIMITED** | Complete run; Uncertain-heavy + Spec=0 on binary subset |
| LLaVA / Gemma A1@1000 | **VALID_BUT_LIMITED** | Pipeline OK; **collapsed** — do not use Acc/F1 as skill |
| InternVL3 Stage 1 | **FAIL_TECHNICAL** | Not A1-1000-qualified |
| Phi-4 A2@1000 V100 / MiniCPM·Molmo L40S cancel | **SUPERSEDED_RUNTIME** | Replaced by later successful jobs |

---

## 5. What to keep vs archive

### A. PRIMARY_RESULTS

- `outputs/verification_dataset/` + `verification_ablation_*`
- Qwen2.5 `20260706_2214` (A1–A5 @1000) + `20260708_0020` (**A1–A5 @5747**)
- Qwen3-VL `qwen3vl_A1A5_5747` (A1–A5 @5747)
- GLM `20260919_glm46v_flash_A1A5_5747` (A1–A5 @5747)
- Phi-4 `20260921_phi4_A1A5_5747` (A1–A5 @5747)
- Advisor comparison: [`docs/FULL_SCALE_MODEL_COMPARISON.md`](FULL_SCALE_MODEL_COMPARISON.md)
- Qwen-only detail: [`docs/QWEN_FULL_A1_A5_RESULTS.md`](QWEN_FULL_A1_A5_RESULTS.md)
- Cross-model A1@1000 audit under `outputs/analysis/a1_1000_cross_model_audit/`

### B. SUPPORTING / QUALIFICATION

- LLaVA/Gemma A1-1000 (**collapse evidence only**)
- Phi-4 / GLM / Qwen3 / MiniCPM / Molmo qualification smokes and @1000 runs
- MiniCPM & Molmo A1@5747 (**A1 only** evidence, not peer A1–A5)

### C. ARCHIVED / SUPERSEDED

See `archive/ARCHIVE_MANIFEST.csv`. Includes failed Qwen pilots, duplicate runs, smokes moved to archive, etc.

---

## 6. Remaining open items

1. **MiniCPM / Molmo A2–A5 @5747** — **Not evaluated**; A1 evidence suggests limited verification value (near-collapse / Spec≈0). Decide deliberately before more GPU.
2. **LLaVA / Gemma** — **Collapsed** under current protocol; retained as negative controls only.
3. **InternVL3** — **Qualification failed / stopped**; not A1-1000-qualified until Stage 1 is re-passed.
4. Metrics JSON **Reliable%** on some older partial-N runs used a 5747 denominator quirk — prefer counts from this audit / comparison doc for full-scale cells.
5. Optional: publication figures / deeper error analysis across the four complete models.

**Resolved (historical blockers):** Qwen2.5 A5 @5747 complete; Phi-4 A2@1000 clean rerun complete; Phi-4 A1–A5 @5747 complete; Qwen3-VL A1–A5 @5747 complete; MiniCPM & Molmo A1@5747 complete on H200.

---

## 7. Recommended next research action

**Document and present** the four-model A1–A5 @5747 comparison ([`FULL_SCALE_MODEL_COMPARISON.md`](FULL_SCALE_MODEL_COMPARISON.md)). Treat MiniCPM/Molmo as **A1 only** negative controls unless a scientific reason justifies A2–A5.

---

## 8. Historical A1@1000 audit notes (2026-09-09)

The following sections preserve the original cross-model audit detail for LLaVA/Gemma collapse diagnosis. They are **not** the current full-scale status.

### Prediction distributions (A1-1000)

| Model | Reliable | Uncertain | Unreliable | Collapse? |
|-------|----------:|----------:|-----------:|-----------|
| **Qwen** | 672 (67.2%) | 272 (27.2%) | 56 (5.6%) | No |
| **LLaVA** | **1000 (100%)** | 0 | 0 | **Yes** |
| **Gemma** | **1000 (100%)** | 0 | 0 | **Yes** |

### Pipeline / input parity (unchanged conclusions)

- Positive prediction label: **`Reliable`**; **Uncertain excluded** from binary metrics.
- Parser does **not** default malformed output to Reliable.
- LLaVA/Gemma collapse is **model behavior**, not missing prompts/images or swapped GT.
- On 72 GT-negatives: LLaVA & Gemma → **72 Reliable** (all FP).

### InternVL3 Stage 1 (balanced 100)

Predictions: Reliable 56, Uncertain 31, Unreliable 0, parse-fail 13. Spec=**0**; BalAcc=**0.5**. **Not qualified for A1-1000.**

---

## End summary

**Complete (A1–A5 @5747):** Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4.

**A1 only @5747:** MiniCPM-V-4.5, Molmo2-8B (H200).

**Collapsed (do not use Acc/F1 as skill):** LLaVA A1@1000, Gemma A1@1000.

**DO NOT RERUN** completed verified full-scale cells above unless protocol changes.
