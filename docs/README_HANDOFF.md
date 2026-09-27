# README Handoff — Evidence Dossier

**Purpose:** Factual package for another agent to redesign `README.md`.  
**Not a README redesign.** Do **not** treat this file as user-facing prose.

> **2026-09-27 update:** This dossier was written while InternVL3.5 was still running and before Gemma 4 12B IT was run. Stale statements (§6.2, §6.6, §6.7, §8, §9, §13, §15, §16, §17 and fact-sheet items 10, 19, 21, 28) have been corrected in place. For current status always use [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md): InternVL3.5 and Gemma 4 are both **Complete** A1–A5 @5747 and both **collapsed** (abstention-heavy and Reliable-heavy respectively).

**Audit timestamp (this handoff):** 2026-09-24 ~17:25 America/New_York  
**Auditor constraints:** no inference, no Slurm submit/cancel, no model download, no `README.md` edits, no prompt/parser/evaluator edits, no commit/push.

**Primary sources of truth (prefer these over current README):**

| Doc / tree | Role |
|------------|------|
| `docs/EXPERIMENT_STATUS_CANONICAL.md` | Inventory, jobs, collapse labels |
| `docs/FULL_SCALE_MODEL_COMPARISON.md` | Advisor A1–A5 @5747 tables |
| `docs/INTERNVL3_5_HF_QUALIFICATION.md` | InternVL3 vs 3.5 gate |
| `docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md` | Collapse forensics + expansion shortlist |
| `docs/EVALUATION_PROTOCOL.md` | Matching + binary semantics |
| `docs/ABLATION_STUDY.md` | A1–A5 design |
| `outputs/evaluation/**/**/A*_metrics.json` | Authoritative numeric metrics |
| `outputs/verification/**` | Authoritative prediction trees |
| Live Slurm `squeue` + on-disk sample counts | InternVL3.5 progress **now** |

---

## 1. PROJECT ONE-LINER

Second-stage open-weight VLM verification of **fixed YOLO wild-palm detections** on UAV orthomosaic patches: each detection is labeled **Reliable / Uncertain / Unreliable** under controlled visual-context ablations **A1–A5**, then scored against LabelMe `palm` ground truth. This repository does **not** run palm detection as its scientific product; YOLO detections are frozen inputs.

**Thesis framing (existing):** CS Honors Thesis · Annie Luo · Mentor Fan Yang · Wake Forest University · 2026.  
Source: current `README.md` header (author metadata only).

---

## 2. SCIENTIFIC QUESTION

Exact wording from current README (matches experimental intent in canonical docs):

> Can modern VLMs verify YOLO palm detections by accepting true positives and rejecting false positives, and how does available visual context (ablations **A1–A5**) change verification reliability?

Closed-set decisions: **Reliable** (accept) · **Uncertain** (abstain) · **Unreliable** (reject). Compared models share frozen dataset, prompts, parser, and evaluator.

Source: `README.md` Research Question; protocol shared with `docs/FULL_SCALE_MODEL_COMPARISON.md` §A, `docs/EVALUATION_PROTOCOL.md`.

---

## 3. PIPELINE

Scientifically correct end-to-end flow (VLM does **not** detect):

```
UAV orthomosaic patches
  → YOLO detection (frozen; outside/verification-framework product)
  → verification_dataset (one overlay sample per YOLO box)
  → A1–A5 ablation construction (same boxes; only VLM inputs change)
  → VLM verifier (model-native vision/chat + shared JSON parser)
  → decision ∈ {Reliable, Uncertain, Unreliable}
  → LabelMe GT matching (greedy 1–1, IoU ≥ 0.5)  [evaluation only]
  → binary + distribution metrics
```

**Critical clarifications:**

- LabelMe GT is used for **evaluation**, not as VLM input.
- Matching/GT polarity is computed from YOLO boxes vs LabelMe `palm` boxes.
- Frozen shared stack: prompts template family, parser, evaluator, ablation definitions.
- Model-specific: adapter + YAML config + registry entry + checkpoint load path.

Sources: `docs/ARCHITECTURE.md` Diagram 1; `docs/EVALUATION_PROTOCOL.md`; `docs/ABLATION_STUDY.md`; `src/preprocessing/verification_overlay.py`; `src/verification/registry.py`.

---

## 4. CANONICAL DATASET / EVALUATION

### Dataset (verified)

| Item | Value | Source |
|------|------:|--------|
| N (production detections) | **5747** | `outputs/verification_dataset/`; metrics JSON `samples` / `ground_truth_*` |
| GT+ (matched) | **4685** | e.g. `outputs/evaluation/qwen/20260708_0020/A1/A1_metrics.json` |
| GT− (unmatched) | **1062** | same |
| Always-Reliable Acc prior @5747 | **≈ 0.815** (= 4685/5747) | `docs/EXPERIMENT_STATUS_CANONICAL.md` |
| Always-Reliable Acc prior @1000 | **0.928** (= 928/1000) | same (A1@1000 slice) |

### Matching rule (exact)

- Per image/patch: all YOLO detections × all GT `palm` boxes.
- Pairwise IoU; sort descending; **greedy one-to-one**.
- Accept match iff **IoU ≥ 0.5**.
- Matched detection → GT+; unmatched → GT−.

Source: `docs/EVALUATION_PROTOCOL.md` §2–3; `docs/QWEN_FULL_A1_A5_RESULTS.md` §3.

### Binary evaluation semantics (exact)

| Prediction | Binary role |
|------------|-------------|
| Reliable | positive |
| Unreliable | negative |
| Uncertain | **excluded** from Acc / Prec / Sens / Spec / F1 |

Confusion (binary subset only):

- TP = GT+ ∧ Reliable  
- FP = GT− ∧ Reliable  
- FN = GT+ ∧ Unreliable (**Uncertain ≠ FN**)  
- TN = GT− ∧ Unreliable (**Uncertain ≠ TN**)

Sensitivity = Recall = TP/(TP+FN); Specificity = TN/(TN+FP); BalAcc = (Sens+Spec)/2.  
**Note:** Spec / BalAcc are often **derived** in advisor docs; many `*_metrics.json` files store TP/FP/TN/FN + Acc/Prec/Recall/F1 + R/U/Ur counts but **not** Spec/BalAcc fields.

Source: `docs/EVALUATION_PROTOCOL.md` §3–4; `docs/QWEN_FULL_A1_A5_RESULTS.md` §3; sample metrics JSON under `outputs/evaluation/`.

### Post-hoc descriptive rates (NOT frozen evaluator fields)

Defined in current README and selection analysis as **post-hoc** descriptors derived from existing R/U/Ur counts:

| Rate | Definition |
|------|------------|
| **Decision Coverage** | (Reliable + Unreliable) / N |
| **Abstention Rate** | Uncertain / N |

Do **not** claim these are written by the canonical evaluator. They are documentation/analysis additions after Molmo2 showed Acc/F1 can look strong on a tiny scored subset.

Source: current `README.md`; `docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md` §0 proposed coverage gate.

---

## 5. EXACT A1–A5 DEFINITIONS

Canonical names in code (`src/paths.py`, `src/prompts/ablation_verification_prompts.py`):

| Code key | Doc short name | Image input | Prompt metadata |
|----------|----------------|-------------|-----------------|
| `A1_overlay_only` | **A1 — Overlay Only** | Overlay image only | None |
| `A2_overlay_confidence` | **A2 — Overlay + Confidence** | Overlay | YOLO confidence |
| `A3_overlay_confidence_geometry` | **A3 — Overlay + Confidence + Geometry** | Overlay | Confidence + bbox geometry (area, width, height, aspect ratio, center) |
| `A4_overlay_crop_confidence` | **A4 — Dual Panel (Overlay + Crop) + Confidence** | Combined dual-panel (full overlay + enlarged bbox crop) | YOLO confidence |
| `A5_crop_only` | **A5 — Crop Only + Confidence** | Enlarged bbox crop only (no surround) | YOLO confidence; no overlay; no geometry |

**Overlay construction (code):** dim outside bbox by `DEFAULT_DIM_FACTOR = 0.55`, restore bbox region, draw target box (`src/preprocessing/verification_overlay.py`).

**Image storage fact:**

- A1–A3 `prompt_index.csv` points at `outputs/verification_dataset/images/*.png` (shared overlays).
- A4–A5 store dedicated images under `outputs/verification_ablation_<N>/A4_*/images/` and `A5_*/images/`.

Sources: `docs/ABLATION_STUDY.md`; `docs/FULL_SCALE_MODEL_COMPARISON.md` §A; `src/preprocessing/verification_overlay.py`; ablation `prompt_index.csv` files.

**Do not use** older archive labels `E1_raw_crop` / `E2_bbox_only` as current A1–A5 (`archive/superseded_runs/...`).

---

## 6. COMPLETE MODEL INVENTORY

Coverage / Abstention below are **computed post-hoc** from R/U/Ur where N is known. Spec / BalAcc from canonical docs / derived from TP/FP/TN/FN when noted.

### 6.1 Full-scale functional (Complete A1–A5 @5747)

| Model | Checkpoint / ID | Family / API (documented) | Largest run | Status | A1 R/U/Ur | A1 Acc | A1 Sens | A1 Spec | A1 BalAcc | A1 F1 | A1 Coverage | A1 Abstention | Collapse | Caveats |
|-------|-----------------|---------------------------|-------------|--------|-----------|-------:|--------:|--------:|----------:|------:|------------:|--------------:|----------|---------|
| Qwen2.5-VL-7B-Instruct | `…/models/Qwen2.5-VL-7B-Instruct` (legacy tree `outputs/verification/qwen/`) | Qwen2.5-VL HF | Full A1–A5 @5747 `20260708_0020` | **Complete** | 3593/1775/379 | 0.8512 | 0.9381 | 0.3059 | ≈0.622 | 0.9158 | 0.691 | 0.309 | No | PRIMARY; three-way labels |
| Qwen3-VL-8B-Instruct | `…/models/Qwen3-VL-8B-Instruct` | Qwen3-VL HF | Full A1–A5 @5747 `qwen3vl_A1A5_5747` | **Complete** | 4309/246/1192 | 0.7519 | 0.8273 | 0.4138 | ≈0.621 | 0.8450 | 0.957 | 0.043 | No | SUPPORTING within-family; A5 Uncertain-heavy |
| GLM-4.6V-Flash | `…/models/GLM-4.6V-Flash` | GLM local | Full A1–A5 @5747 `20260919_glm46v_flash_A1A5_5747` | **Complete** | 3719/50/1978 | 0.6709 | 0.6983 | 0.5492 | ≈0.624 | 0.7760 | 0.991 | 0.009 | No | PRIMARY; strong Unreliable usage |
| Phi-4 Multimodal | `…/models/Phi-4-multimodal-instruct` | Phi-4 remote-code | Full A1–A5 @5747 `20260921_phi4_A1A5_5747` | **Complete** | 4985/0/762 | 0.7661 | 0.8886 | 0.2260 | ≈0.557 | 0.8610 | 1.000 | 0.000 | No (0 Uncertain) | PRIMARY; effectively binary verifier |

Sources: `docs/FULL_SCALE_MODEL_COMPARISON.md` §C–D; `docs/EXPERIMENT_STATUS_CANONICAL.md` §1; metrics under `outputs/evaluation/{qwen,qwen3_vl,glm_4_6v_flash,phi4_multimodal}/…`.

### 6.2 Full-scale Complete but collapsed (A1–A5 @5747)

| Model | Checkpoint | Run status | A1 R/U/Ur | A1 Acc | A1 Spec | A1 BalAcc | A1 F1 | A1 Coverage | Outcome |
|-------|------------|-----------|-----------|-------:|--------:|----------:|------:|------------:|---------|
| InternVL3.5-8B-HF | `OpenGVLab/InternVL3_5-8B-HF` → `…/models/InternVL3_5-8B-HF` | **Complete** (`20260924_internvl3_5_hf_A1A5_5747`, jobs 8351993–8351997) | 3085/2570/92 | 0.8898 | 0.1436 | 0.5647 | 0.9407 | 0.553 | **Abstention-heavy collapse** (U 39–71%, Spec ≤ 0.144 across A1–A5) |
| Gemma 4 12B IT | `google/gemma-4-12B-it` → `…/models/gemma-4-12B-it` | **Complete** (`20260925_gemma4_A1A5_5747`, jobs 8353160–8353164) | 5583/0/164 | 0.8156 | 0.0782 | 0.5304 | 0.8968 | 1.000 | **Reliable-heavy collapse** (R 91.5–97.4%, Spec ≤ 0.236); skipped normal qualification gates |

Full per-condition tables: `docs/FULL_SCALE_MODEL_COMPARISON.md` §C–D.

### 6.3 Qualification only / technical failure

| Model | Checkpoint | Run | Status | Key numbers | Classification |
|-------|------------|-----|--------|-------------|----------------|
| InternVL3-8B-Instruct | `…/models/InternVL3-8B-Instruct` | QUAL-100 `20260909_internvl3_qual` | **Qualification failed** | Parse 87% (13 fails); R/U/Ur=56/31/0; Spec=0; BalAcc=0.50; F1=0.769 | **TECHNICAL_FAILURE** (+ Spec0) |

Source: `docs/INTERNVL3_QUALIFICATION.md`; `docs/INTERNVL3_5_HF_QUALIFICATION.md` §2.

### 6.4 Partial collapse (A1 @5747 only; A2–A5 Not evaluated)

| Model | Checkpoint | Run | R/U/Ur | Acc | Sens | Spec | BalAcc | F1 | Coverage | Abstention | Class |
|-------|------------|-----|--------|----:|-----:|-----:|-------:|---:|---------:|-----------:|-------|
| MiniCPM-V-4.5 | `openbmb/MiniCPM-V-4_5` → `…/MiniCPM-V-4_5` | `20260923_minicpm_A1A5_5747` A1 (job 8351080 H200) | 5557/0/190 | 0.8110 | 0.9772 | **0.0782** | **0.5277** | 0.8940 | 1.000 | 0.000 | **PARTIAL_COLLAPSE** Reliable-heavy |
| Molmo2-8B | `allenai/Molmo2-8B` → `…/Molmo2-8B` | `20260923_molmo2_A1A5_5747` A1 (job 8351081 H200) | 1209/4537/1 | 0.9264 | 0.9991 | **0.0000** | ≈0.50 | **0.9618** | **0.211** | **0.789** | **PARTIAL_COLLAPSE** abstention-heavy |

Sources: `docs/EXPERIMENT_STATUS_CANONICAL.md` §5–6; metrics JSON under `outputs/evaluation/minicpm_v4_5/…` and `molmo2_8b/…`.

### 6.5 Single-class collapse (A1 @1000)

| Model | Checkpoint | Run | R/U/Ur | Acc | Spec | F1 | Class |
|-------|------------|-----|--------|----:|-----:|----:|-------|
| LLaVA-OneVision | `…/llava_onevision` (HF `llava-hf/llava-onevision-qwen2-7b-ov-hf` in older docs) | A1@1000 `20260719_1734` | **1000/0/0** | 0.928 | **0** | 0.963 | **SINGLE_CLASS_COLLAPSE** |
| Gemma 3 12B IT | `…/gemma-3-12b-it` | A1@1000 `20260802_1702` | **1000/0/0** | 0.928 | **0** | 0.963 | **SINGLE_CLASS_COLLAPSE** |

Source: `docs/EXPERIMENT_STATUS_CANONICAL.md` §4; `docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md`.

### 6.6 Configured but never run

None. The former `gemma4.yaml` E4B stub was replaced by the Gemma 4 **12B IT** integration, which was run (§6.2). Llama-3.2-11B-Vision-Instruct is the next candidate and has **no** config or adapter.

### 6.7 Registry keys that are integrated (adapters exist)

From `src/verification/registry.py`:  
`qwen2_5_vl`, `qwen3_vl`, `llava`, `gemma`, `gemma4`, `internvl3`, `internvl3_5_hf`, `glm_4_6v_flash`, `phi4_multimodal`, `molmo2_8b`, `minicpm_v4_5` (+ aliases).

---

## 7. FULL-SCALE RESULTS (Complete models only)

Authoritative advisor table recovered from `docs/FULL_SCALE_MODEL_COMPARISON.md` §C–D (verified against on-disk eval trees in consolidation). **Do not invent missing Spec cells** — Spec is listed as in that doc.

### 7.1 Binary metrics @5747

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

Experiment roots:

| Model | Verification / evaluation |
|-------|---------------------------|
| Qwen2.5-VL | `outputs/verification/qwen/20260708_0020/` · `outputs/evaluation/qwen/20260708_0020/` |
| Qwen3-VL | `outputs/verification/qwen3_vl/qwen3vl_A1A5_5747/` · `…/evaluation/qwen3_vl/…` |
| GLM | `outputs/verification/glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747/` |
| Phi-4 | `outputs/verification/phi4_multimodal/20260921_phi4_A1A5_5747/` |

### 7.2 Prediction distributions @5747

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

Source: `docs/FULL_SCALE_MODEL_COMPARISON.md` §D.

---

## 8. LATEST INTERNVL3.5 STATE (re-audited 2026-09-27)

Jobs `8351993`–`8351997` all **COMPLETED** (2026-09-24/25); `squeue -u luoz23` is empty. Experiment `20260924_internvl3_5_hf_A1A5_5747`: **5747/5747** records in every condition, all index rows `ok`, 0 parse / 0 inference errors, evaluation + metrics JSON present for A1–A5.

| Cond | R / U / Ur | Acc | Spec | BalAcc | F1 | Coverage |
|------|-----------|----:|-----:|-------:|---:|---------:|
| A1 | 3085 / 2570 / 92 | 0.8898 | 0.1436 | 0.5647 | 0.9407 | 0.5528 |
| A2 | 3497 / 2239 / 11 | 0.9051 | 0.0265 | 0.5129 | 0.9500 | 0.6104 |
| A3 | 3420 / 2298 / 29 | 0.9063 | 0.0651 | 0.5314 | 0.9505 | 0.6001 |
| A4 | 3343 / 2399 / 5 | 0.9065 | 0.0157 | 0.5079 | 0.9509 | 0.5826 |
| A5 | 1515 / 4078 / 154 | 0.8478 | 0.1269 | 0.5188 | 0.9167 | 0.2904 |

**Verdict for README authors:** InternVL3.5 full-scale is **Complete** and is an **abstention-heavy collapse**. Its Acc/F1 are the highest in the benchmark only because Uncertain is excluded and Unreliable is nearly absent — never quote them without Spec / Coverage. It skipped A1@1000 (balanced-100 PASS → full scale).

---

## 9. KEY SCIENTIFIC FINDINGS

### OBSERVATIONS (evidence-supported)

1. **Six models completed full A1–A5 @5747;** four are useful verifiers (Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4) and two collapsed (InternVL3.5 abstention-heavy; Gemma 4 12B IT Reliable-heavy).  
   Source: `docs/FULL_SCALE_MODEL_COMPARISON.md` §B–E.

2. **Context sensitivity is large within a model.** Example: Qwen2.5 Spec A3=**0.0179** vs A5=**0.7198**; Qwen3 A5 R/U/Ur=**1081/3948/718** with Spec=**0.8121**.  
   Source: full-scale tables §7.

3. **Acceptance-heavy collapse exists.** LLaVA & Gemma A1@1000 → 100% Reliable; Acc=class prior 0.928; Spec=0. MiniCPM A1@5747 → R=5557, Spec=0.0782, Acc≈0.811≈prior.  
   Source: canonical inventory §4–5.

4. **Abstention-heavy collapse exists.** Molmo2 A1@5747 → U=4537 (78.95%); Spec=0 on binary subset; F1=0.9618 on only ~1210 scored predictions.  
   Source: Molmo metrics JSON + canonical §6.

5. **Rejection / specificity differs across functional models.** GLM uses substantial Unreliable (~30–44%) with Spec ~0.55–0.76; Phi-4 never abstains (U=0 all conditions) but still rejects via Unreliable; Qwen2.5 Spec collapses under A3 metadata.  
   Source: §7 distributions + FULL_SCALE §E.

6. **Binary Acc/F1 alone are unsafe.** Always-Reliable prior @5747 ≈0.815; Molmo2 F1 high despite Spec=0; LLaVA Acc=0.928 with Spec=0.  
   Source: protocol + Molmo/LLaVA evidence.

7. **InternVL3 → InternVL3.5 (same balanced-100 gate):** parse 87%→100%; Spec 0→0.20; BalAcc 0.50→0.60; Uncertain-heavy (58/100). Full scale (Complete): abstention-heavy collapse, Spec ≤ 0.144.  
   Source: `docs/INTERNVL3_5_HF_QUALIFICATION.md`.

### CAUSAL INTERPRETATIONS (not proven — do not state as fact in README)

- That dimming/overlay “causes” success/failure.
- That models “ignore the image.”
- That InternVL3.5 improvement is due **only** to model version (API also changed).
- That Uncertain equals calibrated uncertainty.
- That any untested model will avoid collapse.

Sources for the caution: `docs/INTERNVL3_5_HF_QUALIFICATION.md` §5; `docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md` claims sections.

---

## 10. FAILURE / BEHAVIOR TAXONOMY

| Mode | Operational definition | Evidence |
|------|------------------------|----------|
| **Single-class affirmative collapse** | ≥95% one class (esp. Reliable); Acc≈prior; Spec≈0; parser OK | LLaVA / Gemma A1@1000 = 1000/0/0 |
| **Reliable-heavy partial collapse** | Multi-label possible but near-always-Reliable; Spec≈0; Acc≈prior | MiniCPM A1@5747: 5557/0/190; Spec=0.0782; Acc=0.811 |
| **Abstention-heavy partial collapse** | Uncertain dominates; Unreliable≈absent; Spec=0 on scored subset | Molmo2 A1@5747: 1209/4537/1; Spec=0; Coverage≈0.211 |
| **Structured-output / parse failure** | Non-empty text fails JSON parse | InternVL3 Stage1: 13/100 parse fails |
| **Functional trade-off (non-collapse)** | Non-trivial Spec or three-way structure; condition-dependent Sens/Spec | Qwen / GLM / Phi full matrix; e.g. Qwen2.5 A3 high F1 / near-zero Spec |

Source: `docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md` §0 taxonomy; canonical status docs.

---

## 11. MOLMO2 CASE (exact verified numbers)

From `outputs/evaluation/molmo2_8b/20260923_molmo2_A1A5_5747/A1/A1_metrics.json` and canonical §6:

| Field | Value |
|-------|------:|
| N | 5747 |
| Reliable / Uncertain / Unreliable | **1209 / 4537 / 1** |
| Uncertain % | **78.95%** |
| Binary scored n | TP+FP+TN+FN = 1121+88+0+1 = **1210** |
| Acc / Prec / Sens / Spec / F1 | 0.9264 / 0.9272 / 0.9991 / **0.0000** / **0.9618** |
| TP / TN / FP / FN | 1121 / **0** / 88 / 1 |
| Decision Coverage | (1209+1)/5747 ≈ **0.211** |
| Abstention Rate | 4537/5747 ≈ **0.789** |
| Parse / inference failures | **0 / 0** |
| Classification | **PARTIAL_COLLAPSE** (abstention-heavy) |

**README-safe explanation:** F1=0.962 is computed **only** on the ~21% of predictions that are Reliable/Unreliable. Uncertain is excluded by protocol, so high F1 does **not** mean strong verification skill when Spec=0 and Unreliable≈absent.

---

## 12. INTERNVL3 → INTERNVL3.5 CASE

Same gate: `outputs/diagnostics/model_qualification/balanced_A1_100/` (50/50).

| | InternVL3-8B-Instruct | InternVL3.5-8B-HF |
|--|----------------------:|------------------:|
| Experiment | `20260909_internvl3_qual` | `20260924_internvl3_5_hf_qual` |
| Implementation | remote-code `AutoModel` + `model.chat` | HF-native `AutoProcessor` + `AutoModelForImageTextToText` |
| Parse success | **87%** (13 fails) | **100%** |
| R / U / Ur | 56 / 31 / 0 | 39 / 58 / 3 |
| TP / TN / FP / FN | 35 / 0 / 21 / 0 | 27 / 3 / 12 / 0 |
| Spec | **0.00** | **0.20** |
| BalAcc | **0.50** | **0.60** |
| Sens | 1.00 | 1.00 |
| F1 | 0.769 | 0.818 |
| Verdict | **FAIL_TECHNICAL** | **PASS** |

**Causal limitation (must keep):** model **version** and **implementation/API** both changed — improvement cannot be attributed to version alone.

Source: `docs/INTERNVL3_5_HF_QUALIFICATION.md` §2, §5.

---

## 13. MODEL EXPANSION (documented direction)

Distinguish status carefully:

### EVALUATED (done)

Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4, InternVL3.5-8B-HF, Gemma 4 12B IT (all A1–A5 @5747), MiniCPM-V-4.5 (A1), Molmo2-8B (A1), LLaVA-OneVision (A1@1000), Gemma 3 (A1@1000), InternVL3 (qual fail).

### RUNNING

None (2026-09-27).

### NEXT CANDIDATE / NOT YET EVALUATED

- **Llama-3.2-11B-Vision-Instruct** — no adapter, config, job, or download.
- Optional plan in docs: **Kimi-VL-A3B-Instruct** qualification Stages 0–1 **NOT EXECUTED**.

### RESEARCHED ONLY (external landscape; **not** experiments)

Documented in `docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md` §0 (2026-09-24 research):

| Candidate | Doc priority | Status in this repo |
|-----------|--------------|---------------------|
| Kimi-VL-A3B-Instruct | Optional #1 if MoE gap remains | **Research + plan only** (no adapter/config/run) |
| NVIDIA Llama-3.1-Nemotron-Nano-VL-8B | Runner-up | Researched only |
| Gemma 4 E4B-it | Low (Gemma 3 collapsed) | Not run; the **12B IT** variant was run instead (§6.2) |
| Aya Vision 8B / SmolVLM2 / DeepSeek-VL2 / Janus / Llama 4 | Low / do-not | Researched only |

**Stopping rule (doc):** finish InternVL3.5; if non-collapsed + distinct profile, primary matrix may be sufficient; at most one optional Kimi; do not expand MiniCPM/Molmo A2–A5 or LLaVA/Gemma without a new question.

---

## 14. REPOSITORY ARCHITECTURE

### Frozen / shared

| Component | Path |
|-----------|------|
| Evaluation protocol | `docs/EVALUATION_PROTOCOL.md`, `scripts/evaluate_verification_against_groundtruth.py`, `scripts/compute_verification_metrics.py` |
| Ablation design + prompts | `docs/ABLATION_STUDY.md`, `src/prompts/ablation_verification_prompts.py`, `src/prompts/ablation_verification_prompt_builder.py` |
| Overlay builder | `src/preprocessing/verification_overlay.py` (`DEFAULT_DIM_FACTOR=0.55`) |
| Verification runner / records / resume | `src/verification/runner.py`, `records.py`, `output_manager.py`, `verification_resume.py` |
| Shared parser | `src/lvm/` parser modules used by adapters (shared JSON decision parse) |
| Paths | `src/paths.py` |

### Model-specific

| Component | Path |
|-----------|------|
| Adapters | `src/lvm/*_verification_adapter.py`, `*_verifier.py` |
| Configs | `configs/models/*.yaml` |
| Registry | `src/verification/registry.py` |

### Runner / Slurm / scripts

| Item | Path |
|------|------|
| Primary submit helper | `scripts/submit_model_ablation.sh` → `jobs/run_verification.slurm` (via `jobs/lib/model_runtime.sh`) |
| Local/CLI verification | `scripts/run_verification.py` |
| Many legacy per-model Slurm scripts | `jobs/run_*.slurm`, `jobs/download_*.slurm` |
| Eval entry | often chained by jobs when `RUN_EVAL=1` |

### Outputs / docs

| Item | Path |
|------|------|
| Predictions | `outputs/verification/<model_key>/<experiment_id>/A1..A5/` |
| Metrics | `outputs/evaluation/<model_key>/<experiment_id>/A1..A5/*_metrics.json` |
| Canonical docs | `docs/EXPERIMENT_STATUS_CANONICAL.md`, `FULL_SCALE_MODEL_COMPARISON.md`, etc. |
| Outputs layout note | `outputs/README.md` |

---

## 15. REPRODUCTION (current recommended workflow)

**Current recommended multi-model path** (from `scripts/submit_model_ablation.sh` header + current README):

```bash
# Dry-run (default)
./scripts/submit_model_ablation.sh qwen2_5_vl \
  --conditions A1,A2,A3,A4,A5 --limit 5747 --ablation-size 5747 \
  --experiment-id my_exp

# Submit
./scripts/submit_model_ablation.sh … --submit
```

**Also documented (older / still valid for Qwen-centric cluster setup):** `docs/cluster_deployment.md` — conda env, `requirements_cluster.txt`, `scripts/run_verification.py` single-condition example, legacy `jobs/submit_qwen_ablation.sh`.

### Qualification ladder — do **not** claim all models followed it identically

Documented modern gate (selection analysis §10 / InternVL docs):

1. Stage 0 (~10 deliberate samples)  
2. Stage 1 balanced-100 (50/50) with parse≥95%, Spec≥0.20, BalAcc≥0.55, not ≥95% single-class  
3. Optional A1@1000  
4. Full A1–A5 @5747 only if needed  

**Historical reality:**

- Early Qwen full A1–A5 predates the InternVL-style Stage0/1 packaging.
- LLaVA/Gemma were diagnosed mainly at A1@1000.
- MiniCPM/Molmo used sanity → smoke → A1@1000 → A1@5747; **stopped** after partial collapse (A2–A5 not run).
- InternVL3/3.5 used explicit Stage0+Stage1 scripts under `jobs/run_internvl3*_stage*.slurm`; InternVL3.5 then **skipped A1@1000**.
- Gemma 4 12B IT skipped Stage 0, balanced-100 and A1@1000: A1@20 → unbalanced A1@100 (95/0/5) → full A1–A5 @5747.

README redesign should say: **preferred** new-model gate is Stage0→Stage1→(A1@1000)→full, **without** claiming every historical model followed that path.

---

## 16. VISUAL ASSET INVENTORY

~23k image files exist on disk (many archive/superseded). Prefer **canonical** assets below for README.

### Priority A — palm overlays / A1–A5 examples (best for landing README)

| Relative path | Depicts | README-suitable? | Suggested section |
|---------------|---------|------------------|-------------------|
| `outputs/verification_dataset/images/sample_000001.png` (and siblings; **5747** total) | A1-style overlay (dimmed surround + restored bbox + box). Used by A1–A3. | **Yes** (pick 1–3 clean examples) | Pipeline / A1 illustration |
| `outputs/verification_ablation_10/A4_overlay_crop_confidence/images/sample_000001.png` | Dual-panel A4 | **Yes** | A1–A5 gallery |
| `outputs/verification_ablation_10/A5_crop_only/images/sample_000001.png` | Crop-only A5 | **Yes** | A1–A5 gallery |
| `outputs/verification_ablation_5747/A4_*/images/`, `A5_*/images/` | Full-scale A4/A5 (5747 each) | Yes, but heavy; prefer `_10` or `_100` for repo README | Ablation figure |
| `outputs/visualization/sample_*_A1_overlay_only.png` (21 files) | Standalone A1 overlays | **Yes** | Qualitative examples |
| `outputs/visualization/20260706_2214/overlay/sample_*_overlay.png` (**50**) | YOLO vs GT overlay with VLM decision (Qwen A1@1000 viz) | **Yes** | Qualitative / results |
| `outputs/visualization/20260706_2214/comparison/sample_*_comparison.png` (**51**) | Multi-panel original / overlay / GT / YOLO / verification | **Yes (hero candidates)** | Methods or results |
| `outputs/visualization/20260706_2214/failure_cases/` (**237**: ~200 Uncertain, 36 FP, 1 FN) | Failure-mode panels | **Yes** (select FP + Uncertain) | Failure modes |
| `outputs/full_inference/overlays/*_overlay.png` (**880**) | YOLO inference overlays on raw patches | Optional | Detection context (clarify **not** VLM output) |
| `outputs/yolo_gt_overlap_full/` (**881**) | YOLO–GT overlap viz | Optional | Matching / GT |
| `data/samples/images/*.png` (**5**) | Small sample raw patches | Optional demo | Tiny examples |
| `demo/frontend/.tmp_sample.png` | Demo temp (same family as verification overlay) | Prefer not for scientific README | Demo only |

**A1–A3 note:** no separate PNG folders under `verification_ablation_*/A1_*`; images are the verification_dataset overlays referenced by `prompt_index.csv`.

### Priority B — analysis plots (not palm imagery)

| Path | Depicts | README-suitable? | Section |
|------|---------|------------------|---------|
| `outputs/analysis/confidence_histogram.png` | YOLO confidence histogram | Maybe (detection, not VLM) | Appendix / detection stats |
| `outputs/analysis/confidence_boxplot.png` | Confidence boxplot | Maybe | Same |
| `outputs/analysis/confidence_tp_fp_comparison.png` | TP/FP confidence compare | Maybe | Same |

**No checked-in confusion-matrix figures** found as PNG/SVG/PDF under non-archive trees during this audit. Matrices exist as numeric TP/FP/TN/FN in metrics JSON / CSVs — plots would need generation.

### Priority C — archive / superseded (avoid as primary README evidence)

| Path | Notes |
|------|-------|
| `archive/superseded_runs/outputs/ablation_inputs_100/E1_raw_crop/*.png` | Old E1/E2 naming — **not** current A1–A5 |
| `archive/superseded_runs/outputs/ablation_inputs_100/E2_bbox_only/*.png` | Same |

### Missing / not found

- Dedicated publication SVG/PDF figure pack in `docs/`
- InternVL3.5 / Gemma 4 full-scale qualitative boards (runs Complete; boards not generated)
- Checked-in multi-model confusion-matrix plot set

---

## 17. README FACT CHECK WARNINGS (current `README.md`)

| Issue type | What in current README | Guidance for redesign agent |
|------------|------------------------|-----------------------------|
| **Resolved 2026-09-27** | InternVL "running" table and "in flight" bullet | README now reports InternVL3.5 and Gemma 4 as Complete + collapsed |
| **Too detailed for landing page** | Job IDs `8351993`–`8351997`, Coverage/Abstention table, full failure taxonomy | Keep in docs; README should summarize + link canonical docs |
| **Misleading if skimmed** | Listing MiniCPM/Molmo under “Models Evaluated” without emphasizing A1-only | Must mark **A1 only / partial collapse**, not peer A1–A5 |
| **Misleading if Acc/F1 featured** | Molmo2 F1≈0.962 without Coverage | Always pair with R/U/Ur or Coverage/Abstention |
| **Unsupported causal language risk** | Avoid “models ignore images” | Stick to observed decision behavior |
| **Historical path overclaim risk** | “Stage 0 → balanced-100 → A1@1000 → full” as universal | Preferred for **new** models; not all historical runs followed it |
| **Gemma 4 naming** | Gemma 4 12B IT is now in the README model table (Complete, Reliable-heavy collapse) | Name the checkpoint (12B IT, not E4B) and keep the gate-deviation note |
| **Research ≠ experiment** | Kimi not in README (good) | Do not list researched candidates as tested |
| **Slight naming inconsistency** | A4 code key `A4_overlay_crop_confidence` vs prose “dual panel” | Both correct; prefer doc phrasing + code key in methods |

---

## 18. DEAC ACKNOWLEDGMENT

**Search result:** No formal WFU DEAC acknowledgment / DOI / citation block was found in repository markdown.

What **does** exist:

- Operational references to the **DEAC cluster** (paths, Slurm, GPU nodes) in `docs/cluster_deployment.md`, qualification docs, selection analysis.
- Raw patches path note: `/deac/csc/yangGrp/cuij/palm/Raw_Patches/` (`docs/cluster_deployment.md`).
- Author line: Wake Forest University (README).

**Do not invent** DEAC acknowledgment wording, DOI, or grant text. If needed, obtain official WFU DEAC acknowledgment language from DEAC / mentor outside this repo.

---

## README AUTHORING FACT SHEET

Safe facts another agent may use **without** re-auditing (still re-check InternVL progress before claiming Complete):

1. This project is **VLM verification of fixed YOLO palm detections**, not VLM detection.
2. LabelMe GT is for **evaluation only**.
3. Decisions are **Reliable / Uncertain / Unreliable**.
4. N = **5747**; GT+ = **4685**; GT− = **1062**; always-Reliable Acc ≈ **0.815**.
5. Matching: greedy 1–1, **IoU ≥ 0.5**.
6. Binary metrics **exclude Uncertain**.
7. Decision Coverage = (R+Ur)/N; Abstention = U/N — **post-hoc**, not frozen evaluator outputs.
8. Ablations **A1–A5** change VLM inputs only; detector/dataset/parser/evaluator frozen.
9. Overlay dims outside box by **0.55**, restores ROI, draws box.
10. **Complete** full A1–A5 @5747: Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4 (useful verifiers); InternVL3.5-8B-HF, Gemma 4 12B IT (collapsed).
11. Qwen2.5 A3 Spec **0.0179**; A5 Spec **0.7198** (context sensitivity).
12. Qwen3 A5 is Uncertain-heavy: **1081/3948/718**.
13. Phi-4 has **0 Uncertain** on all full-scale conditions.
14. GLM uses substantial **Unreliable** and mid–high Spec.
15. LLaVA & Gemma A1@1000: **100% Reliable**, Spec=0 (**single-class collapse**).
16. MiniCPM A1@5747: **5557/0/190**, Spec=0.0782 (**Reliable-heavy partial collapse**).
17. Molmo2 A1@5747: **1209/4537/1**, Spec=0, F1=0.9618 on ~1210 scored samples (**abstention-heavy**).
18. InternVL3 Stage1 **FAIL** (13 parse fails, Spec=0); InternVL3.5 Stage1 **PASS** (Spec=0.20, BalAcc=0.60) — version **and** API changed.
19. InternVL3.5 full-scale is **Complete** (jobs 8351993–8351997) and **abstention-heavy collapsed** (U 39–71%, Spec ≤ 0.144).
20. MiniCPM/Molmo A2–A5 were **Not evaluated** by design after A1 collapse evidence.
21. Gemma 4 **12B IT** is **Complete** A1–A5 @5747 (jobs 8353160–8353164) and **Reliable-heavy collapsed** (R 91.5–97.4%, Spec ≤ 0.236); it skipped the normal qualification gates.
22. Kimi-VL is a **documented optional future candidate**, not an experiment in this repo.
23. Prefer Spec + R/U/Ur over Acc/F1 alone; Acc can equal class prior under collapse.
24. Authoritative numeric tables live in `docs/FULL_SCALE_MODEL_COMPARISON.md` and `outputs/evaluation/**/A*_metrics.json`.
25. Best README visuals: `outputs/visualization/20260706_2214/comparison/`, verification_dataset overlays, A4/A5 ablation_10 images.
26. Primary submit path today: `scripts/submit_model_ablation.sh`.
27. Do **not** invent DEAC acknowledgment text; none found in-repo.
28. Do **not** modify scientific outputs when redesigning README.
29. Qwen2.5 production tree is legacy key `outputs/verification/qwen/20260708_0020/` (not `qwen2_5_vl/...` for that experiment).
30. Thesis framing: Annie Luo · Mentor Fan Yang · Wake Forest · 2026.

---

## Handoff metadata

| Item | Value |
|------|-------|
| File created | `docs/README_HANDOFF.md` |
| `README.md` modified? | **No** (forbidden) |
| Outputs / jobs / adapters modified? | **No** |
| Commit / push? | **No** |
