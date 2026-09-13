# Wild Palm VLM Verification — Research Status
**Updated: September 13, 2026**

This document is an advisor-readable snapshot of project status. Numerical claims are traced to existing repository outputs and the read-only audit of 2026-09-13. No new inference was run for this write-up.

---

## 1. Research Question

Can general-purpose vision-language models (VLMs) act as **reliable second-stage verifiers** of YOLO wild-palm detections in UAV orthomosaic imagery, and **what visual/contextual information** improves verification?

The VLM does **not** perform primary object detection. Detection is fixed; the VLM only judges each candidate box.

**Pipeline**

```text
Aerial imagery
  → YOLO detection
  → candidate bounding box
  → VLM verifier
  → Reliable / Uncertain / Unreliable
  → evaluation against LabelMe ground truth
```

---

## 2. Dataset and Evaluation Protocol

| Item | Value | Status |
|------|------:|--------|
| YOLO detections used for verification | **5,747** | VALID |
| YOLO confidence threshold | **≥ 0.5** | VALID |
| Ground truth | LabelMe palm annotations | VALID |
| Matching | Greedy one-to-one, IoU descending | VALID |
| IoU threshold | **≥ 0.5** | VALID |
| Full-set GT-positive / GT-negative | **4,685 / 1,062** | VALID (canonical) |
| A1-1000 subset GT+ / GT− | **928 / 72** | VALID |
| Always-Reliable accuracy on A1-1000 | **0.928** | VALID |

Evidence: `outputs/verification_dataset/index.csv`, `outputs/analysis/confidence_summary.json`, `outputs/analysis/a1_1000_cross_model_audit/phase3_gt_distribution.json`.

### Frozen evaluation protocol

- Predicted **Reliable** = binary positive  
- Predicted **Unreliable** = binary negative  
- Predicted **Uncertain** = abstention; **excluded** from binary precision / recall / F1 / accuracy  

This protocol is shared across models (`docs/EVALUATION_PROTOCOL.md`).

### Why specificity and balanced accuracy matter

On A1-1000, **92.8%** of samples are GT-positive. A trivial policy that always predicts Reliable achieves accuracy **0.928** and high F1 **without any negative discrimination**. Specificity and balanced accuracy expose that failure mode; accuracy and F1 alone do not.

---

## 3. A1–A5 Ablation Study

Definitions below follow **production code** in `src/prompts/ablation_verification_prompts.py` (preferred over older prose in `docs/ABLATION_STUDY.md` where they disagree). Detector, GT matching, IoU, and parser remain fixed; only the VLM input changes.

| Ablation | Input | Additional information | Research question |
|----------|-------|------------------------|-------------------|
| **A1** `A1_overlay_only` | Dimmed orthomosaic patch + green YOLO box | None (prompt forbids using confidence/geometry) | Is overlay context alone enough to verify? |
| **A2** `A2_overlay_confidence` | Same overlay | YOLO confidence (auxiliary) | Does detector confidence help? |
| **A3** `A3_overlay_confidence_geometry` | Same overlay | Confidence + width / height / area / aspect ratio | Does explicit box geometry help? |
| **A4** `A4_overlay_crop_confidence` | Dual panel: full overlay + enlarged crop | Confidence | Does local crop detail plus scene context help? |
| **A5** `A5_crop_only` | Enlarged crop only | Confidence | Is local appearance enough without scene context? |

**Scientific progression.** A1→A2 adds detector confidence; A2→A3 adds geometry metadata; A4 introduces a dual-panel visual; A5 removes surrounding context. The study isolates how much each cue contributes to verification—not detection.

> **Note:** Overlay draws the green box (no Palm-ID text in the production overlay path). A3 does **not** inject center coordinates into the prompt; the prompt instructs the model not to use center/raw coordinates.

---

## 4. Qwen2.5-VL Results

**Primary successful verifier:** `Qwen2.5-VL-7B-Instruct`  
Checkpoint: `/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct`

### A1–A5 @ 1,000 — **VALID**

Experiment ID: `20260706_2214`  
Slurm job: **8105350** (`logs/slurm/qwen_ablation_20260706_2214.*`)  
Metrics recomputed from `outputs/evaluation/qwen/20260706_2214/{A*}/{A*}_evaluation.csv` on the shared A1-1000 IDs (GT 928+/72−). Uncertain excluded from F1.

| Ablation | Reliable / Uncertain / Unreliable | F1 | Specificity | Balanced Acc. |
|----------|----------------------------------:|---:|------------:|--------------:|
| A1 | 672 / 272 / 56 | 0.947 | 0.242 | 0.587 |
| A2 | 768 / 211 / 21 | 0.968 | 0.063 | 0.519 |
| A3 | 799 / 200 / 1 | **0.976** | **≈ 0** | 0.499 |
| A4 | 672 / 253 / 75 | 0.942 | 0.424 | 0.669 |
| A5 | 571 / 89 / 340 | 0.792 | **0.846** | **0.755** |

**Highlights**

- **A3** has the highest F1 but **specificity ≈ 0** (almost no true Unreliable on GT-negatives). More metadata is not automatically better.  
- **A5** shows the strongest negative discrimination at N=1000 (Spec 0.846, BalAcc 0.755), trading recall for rejection ability.  
- Parse failures on this series: **0**.

### Full dataset @ 5,747 — A1–A4 **VALID**; A5 **INCOMPLETE**

Experiment ID: `20260708_0020`  
Slurm job: **8106459** (`logs/slurm/qwen_ablation_20260708_0020.*`)

| Ablation | R / U / Ur | F1 | Spec | BalAcc | Status |
|----------|-----------:|---:|-----:|-------:|--------|
| A1 | 3593 / 1775 / 379 | 0.916 | 0.306 | 0.622 | VALID |
| A2 | 4268 / 1344 / 135 | 0.927 | 0.105 | 0.543 | VALID |
| A3 | 4416 / 1313 / 18 | 0.933 | 0.018 | 0.508 | VALID |
| A4 | 3570 / 1557 / 620 | 0.903 | 0.433 | 0.666 | VALID |
| A5 | — | — | — | — | **INCOMPLETE** |

> **A5 @5747 is incomplete.** Only **2,585 / 5,747** result JSON files exist under `outputs/verification/qwen/20260708_0020/A5/` (no `results_index.csv`; no evaluation metrics). Cause: Slurm **TIME LIMIT**. Do **not** present A5 full-set results as finished.

---

## 5. Cross-Model Verification Results

Shared A1-1000 inputs: `outputs/verification_ablation_1000/A1_overlay_only/`  
Audit: `outputs/analysis/a1_1000_cross_model_audit/`

| Model | Exp ID | N | Decision mix (A1-1000) | Spec | BalAcc | Acc | Status |
|-------|--------|--:|------------------------|-----:|-------:|----:|--------|
| **Qwen2.5-VL-7B** | `20260706_2214` | 1000 | R672 / U272 / Ur56 | 0.242 | 0.587 | 0.900 | **VALID** (non-collapsed) |
| **LLaVA-OneVision** | `20260719_1734` | 1000 | **R1000 / U0 / Ur0** | **0** | **0.5** | 0.928 | **COLLAPSE** |
| **Gemma 3 12B IT** | `20260802_1702` | 1000 | **R1000 / U0 / Ur0** | **0** | **0.5** | 0.928 | **COLLAPSE** |
| **InternVL3-8B-Instruct** | `20260909_internvl3_qual` | Qual only | See §6 | 0 (Stage 1) | 0.5 | — | **FAILED QUAL** |

**Interpretation**

- **Qwen** produces meaningful three-way decisions and non-zero true-negative structure.  
- **LLaVA** and **Gemma** completed inference with **0 parse failures**, but predicted **Reliable on all 1000** samples. Accuracy **0.928** and F1 **≈ 0.963** equal the always-Reliable baseline on a set with 928/1000 GT positives—they do **not** demonstrate verification skill. Specificity **0** and balanced accuracy **0.5** reveal the collapse.

> Successful inference and high aggregate Acc/F1 do **not** mean a VLM is functioning as a verifier.

---

## 6. InternVL3 Qualification Failure

Checkpoint: `/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct`  
Write-up: `docs/INTERNVL3_QUALIFICATION.md`

### Stage 0 (10 samples) — technical **PASS**

Job **8303119** · `outputs/verification/internvl3/20260909_internvl3_qual/stage0/`

| Reliable | Uncertain | Unreliable | Parse fail |
|---------:|----------:|-----------:|-----------:|
| 4 | 6 | 0 | 0 |

### Stage 1 balanced-100 — **FAILED QUAL**

Job **8303143** · `…/balanced100/`  
Fixed set: 50 GT+ / 50 GT− (`outputs/diagnostics/model_qualification/balanced_A1_100/`, seed `20260908`)

| Reliable | Uncertain | Unreliable | Parse fail | Spec | BalAcc |
|---------:|----------:|-----------:|-----------:|-----:|-------:|
| 56 | 31 | **0** | **13** | **0** | **0.5** |

Parse failures were identical incomplete JSON stubs: `{"decision": "Reliable",}` (trailing comma). Official binary confusion (Uncertain excluded): TP=35, FP=21, TN=0, FN=0.

### Label-semantics probe (ACCEPT / REVIEW / REJECT)

Job **8303173** · `outputs/diagnostics/model_qualification/internvl_label_semantics_20/`

Controlled rename only:

| Original | Probe |
|----------|-------|
| Reliable | ACCEPT |
| Uncertain | REVIEW |
| Unreliable | REJECT |

**Held constant:** images, checkpoint, decision *definitions* (word-for-word), label order, JSON schema fields, generation settings (`do_sample=False`, `max_new_tokens=512`), image-token placement (`<image>\n`), and InternVL `chat` generation path.

**Prompt parity audited and confirmed:** generation-facing prompts differ only by label-token renaming.

| | Original (same 20 IDs) | After rename |
|--|------------------------:|-------------:|
| Reliable / ACCEPT | 10 | **0** |
| Uncertain / REVIEW | 8 | **0** |
| Unreliable / REJECT | 0 | **20** |
| Parse fail | 2 | **0** |

**Conclusion (careful):** Under the current verification protocol, InternVL3-8B-Instruct exhibits **severe label-framing sensitivity** (20/20 REJECT collapse after a pure label rename) and **failed Stage 1 qualification** (no Unreliable class, Spec=0, parser gate failed). This is a negative result for **this checkpoint + protocol**, not a claim that the entire InternVL family is universally incapable of verification.

**No InternVL A1-1000** should be run under the current protocol.

---

## 7. Key Findings So Far

1. **Qwen2.5-VL can perform meaningful second-stage verification** under the frozen protocol (non-collapsed three-way decisions; usable Spec/BalAcc on several ablations).  
2. **Input representation strongly affects verifier behavior** across A1–A5.  
3. **More metadata is not always beneficial** — A3 maximizes F1 while destroying negative rejection (Spec ≈ 0).  
4. **High F1/accuracy can hide single-class collapse** (LLaVA/Gemma Acc = always-Reliable baseline).  
5. **Specificity and balanced accuracy are essential** on imbalanced verification sets.  
6. **LLaVA-OneVision and Gemma 3 collapsed** to 100% Reliable on shared A1-1000.  
7. **InternVL3 failed qualification** and showed strong label-framing sensitivity in a controlled rename probe.  
8. **Model qualification (small balanced gate) should precede** expensive full ablations or A1-1000 for new VLMs.

---

## 8. Experiment Status Matrix

| Model | Qualification | A1 | A2 | A3 | A4 | A5 | Status |
|-------|---------------|----|----|----|----|----|--------|
| Qwen2.5-VL @1000 | — | VALID | VALID | VALID | VALID | VALID | Primary ablation series |
| Qwen2.5-VL @5747 | — | VALID | VALID | VALID | VALID | **INCOMPLETE** | Finish A5 only |
| LLaVA-OneVision | — | **COLLAPSE** | NOT RUN | NOT RUN | NOT RUN | NOT RUN | Negative control; do not expand |
| Gemma 3 12B | — | **COLLAPSE** | NOT RUN | NOT RUN | NOT RUN | NOT RUN | Negative control; do not expand |
| InternVL3-8B | **FAILED QUAL** | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE | Stop under current protocol |
| Other VLMs | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | Pending selection |

Failed/collapsed models do **not** need A2–A5 runs under the current strategy.

---

## 9. Validity / Rerun Status

### GREEN — Valid / Keep

- Production verification dataset (5,747) and shared A1–A5 ablation inputs  
- Qwen A1–A5 @1000 (`20260706_2214`)  
- Qwen A1–A4 @5747 (`20260708_0020`)  
- A1-1000 cross-model audit artifacts  
- LLaVA/Gemma A1-1000 as **collapse evidence**  
- InternVL Stage 0/1 + label probe as **qualification failure evidence**

### YELLOW — Usable with Caveat

- Some Qwen `*_metrics.json` files embed full-dataset denominators on 1000-sample runs — prefer evaluation CSVs / recomputed tables for rates  
- `docs/ABLATION_STUDY.md` vs production prompt code (Palm ID / center coords) — cite code for what actually ran  
- LLaVA/Gemma Acc/F1 — protocol-comparable, **not** skill claims

### RED — Incomplete / Rerun Required

- **Qwen A5 @5747** — 2,585 / 5,747 outputs; TIME LIMIT; no index/metrics  

### GT count discrepancy (resolved for reporting)

| Source | GT+ / GT− |
|--------|-----------|
| Legacy `outputs/evaluation/A1_metrics.json` | 4709 / 1038 |
| **Canonical** (model eval CSVs + `confidence_summary.json`) | **4685 / 1062** |

Use **4685 / 1062** as canonical for the full verification set.

---

## 10. Open Methodological Question

InternVL’s rename probe raises a protocol design question for **cross-model** verification:

**Should decision labels remain semantic** (`Reliable` / `Uncertain` / `Unreliable`),  
**or use neutral output tokens** (e.g. `A` / `B` / `C`) with definitions mapped externally after decoding?

Semantic labels are human-interpretable and match the scientific task, but may induce model-specific verbalizer / framing bias. Neutral tokens may reduce token preference while shifting complexity into post-processing.

> This requires **literature review** (VLM verification, selective prediction / abstention, answer-choice / verbalizer bias) **before** any change to the frozen canonical protocol.  
> **No protocol change is made in this document.**

---

## 11. Next Steps

### Immediate

1. **Finish / resume Qwen A5 @5747** and evaluate with the frozen GT pipeline.  
2. **Literature review** on: VLM second-stage verification; selective prediction / abstention; answer-choice / verbalizer / label-token bias.  
3. **Decide** whether cross-model decision vocabulary stays semantic or becomes neutral (after review).  
4. **Freeze** the cross-model qualification protocol (balanced Stage 0/1 gate, collapse criteria, framing check).

### New model selection

Do **not** treat any new checkpoint as definitively selected.

The next model should:

- come from an **independent** family relative to Qwen / LLaVA / InternVL / Gemma where possible  
- support **image + instruction** verification  
- be practical on a **48GB NVIDIA L40S**  
- support fine-grained / high-resolution visual judgment  
- support reliable structured output  
- provide meaningful **diversity** beyond models already tested  

**Current candidates pending HF/literature review (not selected):** Phi-4-multimodal; Molmo (and similar independent open VLMs). Qwen3-VL remains a within-family successor option, reported separately from “independent” comparisons.

### Qualification strategy

```text
New model
  → small balanced qualification (fixed 50/50 from A1-1000)
  → decision-distribution sanity check
  → parsing check
  → label-framing robustness check
  → PASS → A1–A5 (then larger N if needed)
  → FAIL → preserve as negative result and stop
```

---

## 12. Advisor Summary

The wild-palm VLM verification pipeline is working end-to-end under a frozen GT and metric protocol. **Qwen2.5-VL-7B** provides a successful primary A1–A5 ablation series at N=1000 and A1–A4 at full N=5747, showing that input representation strongly affects verification—and that high F1 can coexist with near-zero specificity (A3). **LLaVA-OneVision** and **Gemma 3** complete A1-1000 inference but collapse to always-Reliable, so Acc/F1 match the class prior and do not indicate verifier skill. **InternVL3-8B** failed balanced qualification (no Unreliable; Spec=0) and, in a prompt-parity-confirmed label rename, collapsed to 20/20 REJECT—revealing label-framing fragility under the current protocol. One primary full-dataset cell remains incomplete (**Qwen A5 @5747**). The next phase is finish that cell, review protocol robustness (semantic vs neutral decision tokens), and only then qualify a carefully chosen independent VLM through a small balanced gate—not an immediate A1-1000 for InternVL or collapsed models.

---

## 13. Evidence / Reproducibility

| Resource | Path |
|----------|------|
| This snapshot | `docs/ADVISOR_RESEARCH_STATUS_20260913.md` |
| Canonical experiment status | `docs/EXPERIMENT_STATUS_CANONICAL.md` |
| Evaluation protocol | `docs/EVALUATION_PROTOCOL.md` |
| Ablation design (prose) | `docs/ABLATION_STUDY.md` |
| Production ablation prompts | `src/prompts/ablation_verification_prompts.py` |
| Collapse / selection analysis | `docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md` |
| InternVL qualification | `docs/INTERNVL3_QUALIFICATION.md` |
| Verification dataset | `outputs/verification_dataset/` |
| Shared A1-1000 inputs | `outputs/verification_ablation_1000/` |
| Confidence / IoU summary | `outputs/analysis/confidence_summary.json` |
| Cross-model audit | `outputs/analysis/a1_1000_cross_model_audit/` |
| Qwen verification / eval | `outputs/verification/qwen/`, `outputs/evaluation/qwen/` |
| LLaVA | `outputs/verification/llava/20260719_1734/A1/`, `outputs/evaluation/llava/20260719_1734/A1/` |
| Gemma | `outputs/verification/gemma/20260802_1702/A1/`, `outputs/evaluation/gemma/20260802_1702/A1/` |
| InternVL Stage 0/1 | `outputs/verification/internvl3/20260909_internvl3_qual/` |
| Label-semantics probe | `outputs/diagnostics/model_qualification/internvl_label_semantics_20/` |
| Qwen Slurm logs | `logs/slurm/qwen_ablation_20260706_2214.*`, `qwen_ablation_20260708_0020.*` |
| LLaVA / Gemma / InternVL logs | `logs/slurm/llava_A1_20260719_1734.*`, `slurm-8167800.out`, `internvl3_stage0_8303119.*`, `internvl3_stage1_8303143.*`, `internvl3_label_semantics_8303173.*` |

**Relevant Slurm job IDs (from logs/docs):** Qwen @1000 **8105350**; Qwen @5747 **8106459**; LLaVA **8140578**; Gemma **8167800**; InternVL download **8303105**; Stage 0 **8303119**; Stage 1 **8303143**; label probe **8303173**.
