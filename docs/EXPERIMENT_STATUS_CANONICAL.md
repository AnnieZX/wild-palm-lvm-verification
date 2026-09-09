# Experiment Status — Canonical Source of Truth

**Date of audit:** 2026-09-09 (DEAC cluster)  
**Scope:** Read-only inspection of cluster outputs + CPU re-evaluation of Gemma A1-1000. No GPU inference. No deletions. Obsolete artifacts archived with manifest.

**Audit artifacts:** `outputs/analysis/a1_1000_cross_model_audit/`

---

## 1. Canonical datasets

| Asset | Path | Count | Status |
|-------|------|------:|--------|
| Production verification detections | `outputs/verification_dataset/` | **5747** images + prompts + metadata | COMPLETE |
| Index | `outputs/verification_dataset/index.csv` | 5747 | COMPLETE |
| Shared ablation inputs | `outputs/verification_ablation_{10,100,1000,5747}/` | A1–A5 prompts complete at each N | COMPLETE |

A1 overlay images resolve via `prompt_index.csv` → `../../verification_dataset/images/…` (0 missing for N=1000).

---

## 2–4. Canonical model runs

### Comparable A1 / 1000-sample triad (same `sample_id` set **and order**)

Confirmed: `results_index.csv` IDs for Qwen, LLaVA, and Gemma are identical to  
`outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv`  
(`sample_000001` … `sample_001000`).

| Model | Experiment ID | Verification | Evaluation | Prompts/inputs | Expected | Generated | OK | Failed | `results_index.csv` | Eval | Metrics |
|-------|---------------|--------------|------------|----------------|---------:|----------:|---:|-------:|:-------------------:|:----:|:-------:|
| **Qwen** | `20260706_2214` | `outputs/verification/qwen/20260706_2214/A1/` | `outputs/evaluation/qwen/20260706_2214/A1/` | `outputs/verification_ablation_1000/A1_overlay_only/` | 1000 | 1000 | 1000 | 0 | yes | yes | yes |
| **LLaVA** | `20260719_1734` | `outputs/verification/llava/20260719_1734/A1/` | `outputs/evaluation/llava/20260719_1734/A1/` | same | 1000 | 1000 | 1000 | 0 | yes | yes | yes |
| **Gemma** | `20260802_1702` | `outputs/verification/gemma/20260802_1702/A1/` | `outputs/evaluation/gemma/20260802_1702/A1/` | same | 1000 | 1000 | 1000 | 0 | yes | yes* | yes* |

\*Gemma evaluation/metrics were **missing** before this audit; generated 2026-09-09 via existing CPU scripts (`evaluate_verification_against_groundtruth.py`, `compute_verification_metrics.py`) on frozen inference JSON. Raw inference was not modified.

### Other canonical Qwen runs

| Run | Path | Role |
|-----|------|------|
| Qwen A1–A5 @1000 (full ablation) | `outputs/verification/qwen/20260706_2214/` | Primary 1000-scale ablation study |
| Qwen A1–A4 @5747 | `outputs/verification/qwen/20260708_0020/` | Primary full-dataset ablations |
| Qwen A5 @5747 | `…/20260708_0020/A5/` | **INCOMPLETE** (2585/5747; TIME LIMIT); no `results_index.csv` |

---

## 5. Exact sample counts (A1-1000)

All three models: **1000/1000** JSON + index rows, status `ok`, zero parse/inference failures in index.

---

## 6. Prediction distributions (A1-1000, raw JSON)

| Model | Reliable | Uncertain | Unreliable | Parse fail | Unique raw texts | Collapse? |
|-------|----------:|----------:|-----------:|-----------:|-----------------:|-----------|
| **Qwen** | 672 (67.2%) | 272 (27.2%) | 56 (5.6%) | 0 | 475 | No |
| **LLaVA** | **1000 (100%)** | 0 | 0 | 0 | 45 | **Yes** (single class) |
| **Gemma** | **1000 (100%)** | 0 | 0 | 0 | 184 | **Yes** (single class) |

Additional signals:

- LLaVA: top raw response repeated **408/1000**; one visual_reasoning prefix covers **879/1000**.
- Gemma: top raw response repeated **193/1000**.
- LLaVA decision == Gemma decision on **all 1000** samples (both always `Reliable`).
- Qwen differs from LLaVA/Gemma on **328** samples.

### Qwen @5747 decision mix (not collapsed)

| Ablation | Reliable | Uncertain | Unreliable |
|----------|----------:|----------:|-----------:|
| A1 | 3593 (62.5%) | 1775 | 379 |
| A2 | 4268 (74.3%) | 1344 | 135 |
| A3 | 4416 (76.8%) | 1313 | 18 |
| A4 | 3570 (62.1%) | 1557 | 620 |

---

## 7. Ground-truth distribution (same 1000 IDs)

GT from frozen protocol: greedy YOLO↔LabelMe match, **IoU ≥ 0.5**, identical across the three evaluation CSVs.

| | Count | % |
|--|------:|--:|
| GT positive (`matched_gt=True`) | **928** | **92.8%** |
| GT negative | **72** | **7.2%** |

### Trivial baselines (predict one class for all 1000)

| Baseline | Accuracy |
|----------|---------:|
| Always **Reliable** | **0.928** |
| Always **Unreliable** | 0.072 |

**Implication:** On this 1000-slice, **92.8% accuracy is achievable with zero visual understanding** by always predicting Reliable. High accuracy alone is not evidence of good verification performance.

---

## 8–9. Confusion matrices & recomputed metrics (A1-1000 only)

Protocol: `Reliable` = predicted positive; `Unreliable` = predicted negative; **`Uncertain` excluded from binary TP/FP/TN/FN** (same as `compute_verification_metrics.py`). Recomputed from evaluation rows for the 1000 IDs (not blindly trusted from JSON).

### Qwen

| | Pred Reliable | Pred Unreliable | Uncertain |
|--|-------------:|----------------:|----------:|
| GT+ | TP **647** | FN **48** | 233 |
| GT− | FP **25** | TN **8** | 39 |

| Metric | Value |
|--------|------:|
| Precision | 0.9628 |
| Recall | 0.9309 |
| F1 | 0.9466 |
| Accuracy (binary n=728) | 0.8997 |
| Specificity | 0.2424 |
| **Balanced accuracy** | **0.5867** |
| Acc if Uncertain never counted correct ( /1000) | 0.655 |

### LLaVA

| | Pred Reliable | Pred Unreliable |
|--|-------------:|----------------:|
| GT+ | TP **928** | FN **0** |
| GT− | FP **72** | TN **0** |

| Metric | Value |
|--------|------:|
| Precision | 0.9280 |
| Recall | 1.0000 |
| F1 | 0.9627 |
| Accuracy | **0.9280** (= always-Reliable baseline) |
| Specificity | **0.0000** |
| **Balanced accuracy** | **0.5000** |

### Gemma

**Identical** confusion matrix and metrics to LLaVA (100% Reliable).

---

## 10. Validity verdicts

| Run | Classification | Evidence |
|-----|----------------|----------|
| Qwen A1-1000 (`20260706_2214`) | **VALID_PRIMARY_RESULT** | Non-degenerate 3-way decisions; uses Uncertain; beats trivial baseline on specificity/FP control; shared inputs; clean parse |
| Qwen A1–A5 @1000 (same exp) | **VALID_PRIMARY_RESULT** | Full ablation completed + evaluated |
| Qwen A1–A4 @5747 (`20260708_0020`) | **VALID_PRIMARY_RESULT** | 5747/5747 ok; diverse decisions; metrics computed |
| Qwen A5 @5747 | **INCOMPLETE** | 2585/5747; TIME LIMIT; no index; empty eval |
| LLaVA A1-1000 | **VALID_BUT_LIMITED** | Pipeline OK, but **100% Reliable** → accuracy = class prior; specificity 0; not usable as competitive verifier performance |
| Gemma A1-1000 | **VALID_BUT_LIMITED** | Same degeneration as LLaVA after CPU eval |
| Smokes / parity / pilots / duplicates | **ENGINEERING_ONLY** or archived | See archive |

### Direct answers

**QWEN**  
- A1-1000: **scientifically valid** primary result.  
- A1–A4 @5747: **scientifically valid** primary results.  
- A5 @5747: not valid until completed.

**LLAVA**  
- A1-1000 is **not meaningful multi-class verification**. High F1/accuracy is **almost entirely class imbalance + always-Reliable**.  
- No evidence of parser silently forcing Reliable (0 parse errors; decisions present).  
- Input parity OK (same prompt files/images). Collapse is **model behavior** (near-duplicate templated “Reliable” justifications), consistent with prior grounding diagnostics—not a swapped-GT bug.

**GEMMA**  
- Same as LLaVA: completed inference is real; **scientifically degenerate** for verification claims.  
- Evaluation now exists (CPU); metrics match the always-Reliable baseline exactly.

---

## 11. Pipeline audit (Phase 5)

Flow: raw text → `normalize_raw_response` (identity) → JSON extract → `normalize_decision` → `decision` field → eval loads `decision` → `matched_gt` via IoU 0.5 → metrics.

| Question | Finding |
|----------|---------|
| Positive prediction? | Label **`Reliable`** |
| Uncertain in binary metrics? | **Excluded** (not negative) |
| Malformed → Reliable? | **No** — parse failure → empty `decision`, status `parse_error` |
| Parser default? | **No** default to Reliable |
| Missing outputs? | Empty label; excluded from evaluated count |
| Index vs JSON sync (A1-1000)? | Matched for all three |
| GT IoU config? | **Identical** (0.5); GT columns equal across model eval CSVs |
| Metric reporting caveat | Existing `Reliable%` in metrics JSON uses **full 5747 denominator** even for 1000-sample runs—do not interpret those % as within-1000 rates |

**Accuracy inflation risks:** (1) class imbalance; (2) excluding Uncertain from binary accuracy can drop hard cases (affects Qwen); (3) citing LLaVA/Gemma accuracy without specificity/balanced accuracy.

---

## 12. Input parity (Phase 6)

- All three jobs consume `outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv` (LLaVA via experiment script; Gemma via `run_verification.py --prompt-index`; Qwen same ablation builder).
- 0 missing images/prompts; 1000 unique prompt hashes (per-sample metadata differs; shared instruction scaffold).
- Each adapter reads the **same prompt file text** and passes the **resolved image path** into its HF chat template (Qwen / LLaVA / Gemma differ in template API only).
- No empty image placeholders in code paths inspected.
- Chat templates include the user text prompt; no alternate system prompt overrides the verification task in adapters.

**Conclusion:** LLaVA/Gemma degeneration is **not** explained by missing/wrong prompt files or missing images on disk.

---

## 13. Sample-level error analysis (summary)

Tables: `outputs/analysis/a1_1000_cross_model_audit/inspection_set.csv`, `disagreements.csv`, `sample_level_decisions_1000.csv`.

- On **72 GT-negatives**: Qwen → 8 Unreliable / 25 Reliable / 39 Uncertain; LLaVA & Gemma → **72 Reliable** (all FP).
- Disagreements (328) are exclusively Qwen ≠ {LLaVA,Gemma}; LLaVA and Gemma never disagree.
- LLaVA/Gemma raw texts are highly repetitive “looks like a palm → Reliable” templates, including on GT-negatives—**defaulting toward Reliable**, not fine-grained rejection.

---

## 14. What to keep vs archive

### A. PRIMARY_RESULTS

- `outputs/verification_dataset/` + `verification_ablation_*`
- Qwen `20260706_2214` (A1–A5 @1000) + evaluations
- Qwen `20260708_0020` A1–A4 @5747 + evaluations
- Audit analysis under `outputs/analysis/a1_1000_cross_model_audit/`
- Canonical Slurm logs: `logs/slurm/qwen_ablation_20260706_2214.*`, `qwen_ablation_20260708_0020.*`, `llava_A1_20260719_1734.*`, root `slurm-8167800.out` (Gemma A1)

### B. SUPPORTING_RESULTS

- LLaVA/Gemma A1-1000 raw outputs + evals (**as negative controls / collapse evidence**)
- `outputs/analysis/` confidence plots; LLaVA `false_positives.csv`
- Docs under `docs/`; `scripts/debug_llava_grounding.py` findings
- Partial Qwen A5 JSONs under `20260708_0020/A5/` (resume seed only)

### C. ARCHIVED (moved; see `archive/ARCHIVE_MANIFEST.csv`)

Failed A5 `1501`, early empty `2130`, pilots `2145`/`2158`, duplicate `1508`, legacy flat eval/results, old LVM input trees, smokes, debug/parity logs, test fixtures, demo pip junk files.

**Not archived:** LLaVA/Gemma A1-1000 (evidence retained in place).

---

## 15. Remaining unresolved issues

1. **Qwen A5 @5747** incomplete (TIME LIMIT); need A5-only resume with walltime ≫ 24h or multi-job resume.
2. **LLaVA/Gemma** not usable as verifiers under current prompt/decoding until behavior changes (prompt redesign, decoding constraints, or different checkpoints)—root cause is model collapse, not eval bugs.
3. Metrics JSON **Reliable% denominator** quirk for partial-N runs.
4. Qwen binary specificity still low (0.24 @1000); Uncertain-heavy—report balanced accuracy / calibration, not accuracy alone.
5. Gemma job script still does not call evaluation automatically (fixed manually this audit).

---

## 16. Exactly ONE recommended next research action

**Resume and finish Qwen A5 @5747 only** (A5-only job, `RESUME=1`, shared `verification_ablation_5747/A5_crop_only`, walltime/chunking sufficient to clear remaining ~3162 samples), then evaluate A5 with the same CPU GT pipeline—**before** investing in further LLaVA/Gemma accuracy runs.

Rationale: Qwen is the only non-degenerate verifier; A1–A4 full-data results are already primary; A5 is the sole missing primary ablation cell.

---

## End summary

**KEEP:**  
Qwen `20260706_2214` (A1–A5 @1000); Qwen `20260708_0020` A1–A4 @5747; production `verification_dataset` + `verification_ablation_*`; cross-model audit under `outputs/analysis/a1_1000_cross_model_audit/`; LLaVA/Gemma A1-1000 raw+eval as **collapse evidence only**.

**ARCHIVED:**  
Failed/superseded Qwen pilots & duplicate `1508`; legacy flat eval/results; old LVM inputs; smokes; debug/parity/fixtures; pip junk — see `archive/ARCHIVE_MANIFEST.csv`.

**INVALID / DO NOT USE (for primary accuracy claims):**  
LLaVA A1-1000 and Gemma A1-1000 **headline accuracy/F1** (equal to always-Reliable baseline; specificity 0). Pipeline itself is not “broken,” but the **scientific claim of verification skill is invalid**.

**STILL NEEDS INVESTIGATION:**  
Complete Qwen A5 @5747; optional follow-up on *why* LLaVA/Gemma collapse (prompt/decoding/model), only after Qwen matrix is finished.
