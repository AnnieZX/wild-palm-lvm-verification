# Model Selection and Majority-Class Collapse Analysis

**Date:** 2026-09-09  
**Scope:** Research and experimental design only. No production inference changes, no Slurm submissions, no model downloads, no GPU inference in this phase.  
**Related:** `docs/EXPERIMENT_STATUS_CANONICAL.md`, `outputs/analysis/a1_1000_cross_model_audit/`

---

## 1. Executive Summary

On the shared A1-1000 verification set (928 GT-positive / 72 GT-negative), **Qwen2.5-VL-7B-Instruct** produces a non-degenerate three-way decision distribution and non-zero specificity. **LLaVA-OneVision (7B, Qwen2 LLM + SigLIP)** and **Gemma 3 12B IT** both predict **Reliable on all 1000 samples**, yielding accuracy = majority baseline (0.928), specificity = 0, balanced accuracy = 0.5.

We previously ruled out sample-ID mismatch, missing images, prompt-file drift, GT/IoU mismatch, and parser defaults that silently map failures to Reliable. The remaining question is **why model behavior diverges under a fixed task**.

**Strongest supported interpretation (not proven causation):**

1. A1 is a **region-centric fine-grained verification** task (overlay + dimmed context + closed-set JSON labels), not generic open VQA.
2. Qwen2.5-VL is a **natively multimodal** model with explicit **dynamic-resolution** and **localization/grounding** design goals; its outputs vary with sample and express Uncertain/Unreliable with image-tied reasoning.
3. LLaVA-OneVision shares a **Qwen2 language backbone** but a **different vision stack** (SigLIP + AnyRes MLP projector) and LLaVA-style instruction tuning; on our disagreements it emits **highly templated “Reliable”** text that echoes prompt palm traits even when Qwen (and GT) reject the detection.
4. Gemma 3 uses SigLIP@896 with a Gemma text backbone; raw responses also assert Reliable with prompt-like morphology language on GT-negatives—consistent with **prior / affirmative collapse**, not calibrated rejection.

**Recommendation:** Screen new models on a **balanced 50/50 diagnostic gate** before A1-1000. Prefer **InternVL3-8B** (independent family, grounding emphasis) as the next comparison model; treat **Qwen3-VL-8B** as Qwen-family successor, not as independent diversity. Do **not** expand LLaVA/Gemma to full ablations until a non-collapse intervention is demonstrated.

---

## 2. Task Characterization

### Pipeline (production)

```
verification_dataset (YOLO dets × LabelMe patches)
  → ablation builder (verification_ablation_N / A1_overlay_only)
  → image: dim outside bbox (×0.55), restore bbox region, draw green box
  → prompt: verify highlighted detection → Reliable|Uncertain|Unreliable JSON
  → adapter (qwen / llava / gemma) reads prompt file + image path
  → verifier chat template + greedy decode (do_sample=false, max_new_tokens=512)
  → parse_verification_response → decision field
  → greedy IoU≥0.5 GT match → binary metrics (Uncertain excluded)
```

### What A1 actually asks

A1 (`A1_overlay_only`) presents a **full orthomosaic patch** where:

- Background outside the YOLO box is **dimmed** (`DEFAULT_DIM_FACTOR = 0.55`).
- The candidate region is restored at full brightness.
- A **green bounding box** is drawn on the candidate.

The prompt forbids detecting new palms; it requires judging whether the **highlighted** detection is a valid wild palm, using crown/frond/texture morphology, and allowing **Uncertain** under occlusion/ambiguity.

### Cognitive / perceptual demands of A1

| Capability | Required? | Notes |
|------------|-----------|-------|
| Object recognition (palm vs canopy) | **Yes** | Core decision |
| Understanding overlaid bbox / highlight | **Yes** | Prompt + green box + dimming |
| Foreground/background discrimination | **Yes** | Dimmed surround vs restored ROI |
| Localization / region attention | **Yes** | Must not score a different palm elsewhere |
| Fine-grained verification | **Yes** | Aerial crowns are subtle vs broadleaf |
| Spatial reasoning | **Partial** | Relative to box; not full scene graph |
| Uncertainty estimation | **Yes** | Explicit Uncertain class |
| Structured instruction following | **Yes** | Closed JSON schema |
| Generic open VQA | **No** | Wrong abstraction |

**Actual visual decision:** *Given this YOLO proposal, is the content inside the highlighted region a morphologically recognizable wild palm, ambiguous vegetation, or a clear non-palm / bad detection?*

---

## 3. Experimental Evidence

### A1-1000 decision distributions

| Model | Reliable | Uncertain | Unreliable | Unique raw (all 1000) |
|-------|----------:|----------:|-----------:|----------------------:|
| Qwen2.5-VL | 672 | 272 | 56 | 475 |
| LLaVA-OneVision | 1000 | 0 | 0 | 45 |
| Gemma 3 12B IT | 1000 | 0 | 0 | 184 |

### Confusion (binary; Uncertain excluded)

| Model | TP | FP | TN | FN | Spec | BalAcc | Acc |
|-------|---:|---:|---:|---:|-----:|-------:|----:|
| Qwen | 647 | 25 | 8 | 48 | 0.242 | 0.587 | 0.900 (n=728) |
| LLaVA | 928 | 72 | 0 | 0 | 0 | 0.5 | 0.928 |
| Gemma | 928 | 72 | 0 | 0 | 0 | 0.5 | 0.928 |

Always-Reliable baseline accuracy = **0.928** (= LLaVA/Gemma).

### Checkpoints used (cluster)

| Model | Path | Architecture | Vision | Language | Weights ~ |
|-------|------|--------------|--------|----------|-----------|
| Qwen | `/deac/.../models/Qwen2.5-VL-7B-Instruct` | `Qwen2_5_VLForConditionalGeneration` | native ViT (patch 14, dynamic pixels) | Qwen2.5 VL-integrated | ~16.6 GB |
| LLaVA | `/deac/.../models/llava_onevision` (HF: `llava-hf/llava-onevision-qwen2-7b-ov-hf`) | `LlavaOnevisionForConditionalGeneration` | **SigLIP** 384 + AnyRes grid | **Qwen2** (hidden 3584, 28 layers) | ~16.1 GB |
| Gemma | `/deac/.../models/gemma-3-12b-it` | `Gemma3ForConditionalGeneration` | **SigLIP** 896, 256 image tokens | Gemma3 text | ~24.4 GB |

### Decoding (our code)

All three adapters: `do_sample=False`, `max_new_tokens=512`. Collapse is **not** explained by temperature sampling differences in our runner (though HF `generation_config.json` defaults differ and should be audited at load time).

### Shared inputs

Same `verification_ablation_1000/A1_overlay_only/prompt_index.csv`; images ~912×912; 0 missing files; identical prompt instruction scaffold.

---

## 4. Why Qwen2.5-VL Works

### CODE EVIDENCE

- Uses `Qwen2_5_VLForConditionalGeneration` + `process_vision_info` / dynamic processor (`min_pixels=3136`, `max_pixels=12845056`).
- Messages: image path + full verification prompt text.
- Same parser as others; 0 parse errors on A1-1000.
- Outputs store diverse `decision` + longer, sample-varying `visual_reasoning`.

### EXPERIMENTAL EVIDENCE

- Uses all three labels; 328 disagreements with LLaVA/Gemma.
- On GT-negatives in disagreement pool, often states missing crown/fronds or canopy-only content.
- 475 unique raw strings / 1000 (vs 45 for LLaVA).

### MODEL-DOCUMENTATION EVIDENCE

Qwen2.5-VL technical report ([arXiv:2502.13923](https://arxiv.org/abs/2502.13923)):

- Native / dynamic resolution; absolute spatial coordinates for boxes/points.
- Explicit emphasis on **object localization, grounding, and scale-aware perception**.
- Native multimodal training (not LLM-only + late projector-only recipe).

### HYPOTHESIS (not proven)

Qwen’s **vision–language stack is trained for region/grounding tasks**, so “judge this highlighted box” is closer to its training distribution than for LLaVA/Gemma. Affirmative bias may still exist (Reliable is modal), but it is **not total collapse**.

---

## 5. LLaVA-OneVision Collapse Analysis

### Exact model

- Cluster: `llava_onevision` ≈ **llava-onevision-qwen2-7b-ov-hf**
- ~7B; **SigLIP** vision (`image_size` 384); **Qwen2** LLM; AnyRes `image_grid_pinpoints` up to 2304.
- LLaVA-OneVision paper: SigLIP + MLP projector + Qwen2; Higher AnyRes multi-crop; broad instruction-following / task-transfer training—not Qwen2.5-VL’s native dynamic ViT grounding stack.

### Why Qwen2 LLM ≠ Qwen2.5-VL performance

Sharing a **language family** does **not** share:

- vision encoder (SigLIP vs Qwen2.5-VL ViT),
- multimodal pretraining / grounding objectives,
- resolution strategy (AnyRes tiles vs native dynamic tokens),
- instruction-tuning mixture.

### Hypothesis board (A1-1000 evidence)

| ID | Hypothesis | Verdict | Evidence |
|----|------------|---------|----------|
| A | Vision encoder differences | **SUPPORTED** (as contributor) | SigLIP+AnyRes vs Qwen2.5-VL native ViT (config + papers) |
| B | Projector / alignment differences | **PLAUSIBLE** | LLaVA MLP projector vs native VL; not isolated experimentally |
| C | Image resolution differences | **PLAUSIBLE** | 384-base AnyRes vs Qwen dynamic high pixel budget; both see ~912 px source |
| D | Weaker region/localization | **PLAUSIBLE** | Docs emphasize task transfer; less absolute-box grounding than Qwen2.5-VL |
| E | Instruction-tuning differences | **PLAUSIBLE** | LLaVA-OV SFT mixture ≠ Qwen2.5-VL; not ablated here |
| F | Synthetic / template verbalization | **SUPPORTED** (behavior) | 45 unique raw / 1000; 8 unique in 60 disagreements; repeated “clear and well-defined…” |
| G | Affirmative / yes-bias | **SUPPORTED** (behavioral) | 100% Reliable; never Uncertain/Unreliable |
| H | Cannot interpret overlay | **WEAK–PLAUSIBLE** | Text *mentions* bbox/highlight, but may be prompt echo |
| I | Insufficient fine-grained evidence | **PLAUSIBLE** | Possible; not isolated from G/F |
| J | Class-prior exploitation | **SUPPORTED** (effect) | Acc = 0.928 = prior; Spec = 0 |
| K | Semantics of “Reliable” | **PLAUSIBLE** | Word may read as “trust YOLO”; not tested with relabeling |
| L | Generation config | **WEAK** | Our code forces greedy; residual config leakage UNKNOWN |
| M | Cannot express uncertainty | **SUPPORTED** (outcome) | 0 Uncertain; may be preference not capability |
| N | Chat template differences | **PLAUSIBLE** | Different apply_chat_template paths; prompt text identical |

### Collapse locus (LLaVA)

From 60 disagreements (Qwen ∈ {Uncertain, Unreliable}, LLaVA = Reliable):

- LLaVA always asserts Reliable; **8 unique raw** texts.
- On GT-negatives, LLaVA still claims clear palm morphology; Qwen often denies palm features.
- Parser always succeeds → collapse is **not** parsing.
- Best description: **generation-time semantic / decision collapse** with **prompt-echoing rationales**, whether or not perception failed earlier (**perceptual vs semantic not fully separable without probes**).

---

## 6. Gemma 3 Collapse Analysis

### Exact model

- `google/gemma-3-12b-it` locally; `Gemma3ForConditionalGeneration`.
- Vision: SigLIP, **896×896**, `image_seq_length` 256.
- Our verifier: `apply_chat_template` with image path + prompt; greedy decode.

### Hypothesis board

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Always Reliable / Spec=0 | **SUPPORTED** | A1-1000 metrics |
| Templated morphology language on GT-neg | **SUPPORTED** | Disagreement audit |
| Same decision as LLaVA on all 1000 | **SUPPORTED** | Exact agreement |
| Truly inspects ROI vs prior | **UNKNOWN** | Mentions “highlighted region” but may echo prompt; no attention probes |
| SigLIP resolution limit hurts fine crowns | **PLAUSIBLE** | Fixed 896 + 256 tokens |
| Instruction tuning lacks rejection/verification | **PLAUSIBLE** | Not proven |
| Chat/decoding bug unique to Gemma | **WEAK** | Same prompt files; greedy path |

**Collapse locus:** Same pattern as LLaVA—**decision always Reliable** with fluent, low-diversity rationales; parsing OK. Whether failure is perceptual or post-perceptual preference remains **UNKNOWN** without controlled probes (e.g., inverted overlays, empty boxes, label-order swaps).

---

## 7. Candidate Model Research

Criteria: open weights, HF/PyTorch, ~2–14B preferred, single-image, region/grounding usefulness, structured output, DEAC L40S (~46GB) feasible, no API-only.

### Required candidates

#### 1. Qwen3-VL-8B-Instruct

| Field | Assessment |
|-------|------------|
| PARAMETERS | ~8B dense |
| VISION | Qwen3-VL native; DeepStack / dynamic resolution (docs) |
| LANGUAGE | Qwen3 family |
| TRAINING | Native multimodal + instruct; strong 2D grounding claims |
| RESOLUTION | Native dynamic / high-res |
| GROUNDING | **Explicit** design goal (relative coords) |
| STRUCTURED OUTPUT | Strong (JSON-friendly instruct) |
| HF / TRANSFORMERS | Yes (`Qwen/Qwen3-VL-8B-Instruct`) |
| TRUST_REMOTE_CODE | Typically false on recent HF |
| VRAM | ~MEDIUM (similar to 7–8B bf16) |
| DEAC | Good; config stub already in repo (`configs/models/qwen3_vl.yaml`) |
| IMPLEMENTATION | LOW–MEDIUM (extend Qwen adapter) |
| SCIENTIFIC VALUE | High for **successor**; **low independence** |
| COLLAPSE RISK | Lower *a priori* than LLaVA/Gemma given grounding lineage; **still must gate** |
| FAMILY | **SAME FAMILY AS QWEN** |

#### 2. InternVL3-8B-Instruct

| Field | Assessment |
|-------|------------|
| PARAMETERS | ~8B |
| VISION | InternViT + dynamic tiling (448 base; multi-tile) |
| LANGUAGE | Qwen2.5-class LLM in InternVL3 line (**partially related**) |
| TRAINING | Native multimodal pretrain + SFT; documents **visual grounding** |
| GROUNDING | Emphasized in model card |
| HF | `OpenGVLab/InternVL3-8B-Instruct` |
| TRUST_REMOTE_CODE | **Often true** (implementation risk) |
| VRAM | MEDIUM |
| DEAC | Feasible on L40S |
| IMPLEMENTATION | MEDIUM–HIGH (custom chat / pixel_values helpers) |
| SCIENTIFIC VALUE | **High** as non-Qwen-*VL* stack with grounding |
| COLLAPSE RISK | Unknown; must gate |
| FAMILY | **PARTIALLY RELATED TO QWEN** (LLM) / **independent VL recipe** |

#### 3. InternVL3-14B-Instruct

Same family as above; ~15B; **HIGH** compute; use only if 8B passes Stage 1.

#### 4. Phi-4-multimodal-instruct

| Field | Assessment |
|-------|------------|
| PARAMETERS | ~5.6B total (3.8B LLM + vision/audio) |
| VISION | SigLIP-400M + MLP + LoRA (Mixture-of-LoRAs) |
| LANGUAGE | Phi-4-Mini |
| RESOLUTION | ~448 + multi-crop HD transform |
| GROUNDING | Not a flagship grounding specialist vs Qwen3/InternVL |
| HF | `microsoft/Phi-4-multimodal-instruct` |
| TRUST_REMOTE_CODE | Likely **true** (custom `modeling_phi4mm`) |
| VRAM | LOW–MEDIUM |
| IMPLEMENTATION | MEDIUM–HIGH |
| SCIENTIFIC VALUE | Independent **Microsoft** stack; good low-cost probe |
| COLLAPSE RISK | **Non-trivial**—SigLIP lineage shared with failed LLaVA/Gemma vision style |
| FAMILY | **INDEPENDENT** |

### Additional candidates (up to 4)

#### 5. InternVL3.5-8B-HF

HF-native export (`OpenGVLab/InternVL3_5-8B-HF`); similar scientific role to InternVL3-8B with potentially cleaner transformers integration. **PARTIALLY RELATED** LLM lineage possible. Prefer if `trust_remote_code` burden drops.

#### 6. MiniCPM-V 2.6 (~8B)

SigLIP + **Qwen2-7B**; strong OCR/mobile claims; **PARTIALLY RELATED**. Useful efficiency probe but **less independent** and SigLIP+Qwen2 pattern resembles LLaVA-OV risk.

#### 7. Kimi-VL-A3B-Instruct (~16B MoE, ~2.8B+0.4B act.)

MoonViT + MoE Moonlight; strong agent/grounding reports. **INDEPENDENT**. Implementation/MoE ops risk MEDIUM; good diversity pick if Stage 0 loads cleanly.

#### 8. Molmo-7B-D (Ai2)

Open weights + open data; **INDEPENDENT**. Solid scientific comparison; verify current transformers support and VRAM before committing.

---

## 8. Candidate Comparison Table

| Model | Family vs Qwen2.5-VL | Grounding fit | Indep. value | Impl. risk | Compute | Screen first? |
|-------|----------------------|---------------|--------------|------------|---------|---------------|
| Qwen3-VL-8B | SAME | Excellent (docs) | Low | Low | MEDIUM | Yes (successor) |
| InternVL3-8B | PARTIAL (LLM) | Strong (docs) | High | Med–High | MEDIUM | **Yes (priority)** |
| InternVL3-14B | PARTIAL | Strong | High | Med–High | HIGH | Only if 8B passes |
| Phi-4-multimodal | INDEPENDENT | Moderate | High | Med–High | LOW–MED | Yes (cheap) |
| InternVL3.5-8B-HF | PARTIAL | Strong | High | Med | MEDIUM | Alt to InternVL3 |
| MiniCPM-V 2.6 | PARTIAL | Moderate | Med | Med | MEDIUM | Optional |
| Kimi-VL-A3B | INDEPENDENT | Strong (docs) | High | Med | MEDIUM | Optional diversity |
| Molmo-7B | INDEPENDENT | Moderate–Strong | High | Med | MEDIUM | Optional |

**Do not rank by MMMU.** Rank by: region verification fit, collapse risk, independence, DEAC feasibility.

---

## 9. Model-Family Independence

Recommended **comparison set** for the thesis:

1. **Qwen2.5-VL-7B** — established non-collapse baseline (already run).
2. **InternVL3-8B-Instruct** — primary **new** comparison (different VL recipe; grounding emphasis).
3. **Phi-4-multimodal** or **Molmo-7B** — independent low/med-cost third family.
4. **Qwen3-VL-8B** — optional **successor** study (same lineage; report separately as “within-family improvement,” not “cross-architecture confirmation”).

Avoid claiming “three independent VLMs succeed” if the only successes are Qwen2.5 + Qwen3.

---

## 10. Qualification Protocol

**Do not run A1-1000 or full ablations until Stage 1 pass.**

### Stage 0 — Sanity (n=10) — LOW

Hand-pick: 4 clear GT+, 4 clear GT−, 2 ambiguous. Check load, image path, JSON parse, ≥2 distinct decisions if obvious negatives exist.

**Fail:** cannot load; empty images; parse fail rate >20%; all 10 identical label when set includes clear GT−.

### Stage 1 — Balanced diagnostic (50 GT+ / 50 GT−) — LOW–MEDIUM

Sample from A1-1000 (or full 5747) **without changing official production metrics**.

**Researcher-defined pass thresholds** (not community standards):

| Metric | Pass if |
|--------|---------|
| Parser success | ≥ 95% |
| Decision diversity | Not ≥95% single class |
| Specificity (binary) | ≥ 0.20 |
| Balanced accuracy | ≥ 0.55 |
| Negative recall (Unreliable∪Uncertain on GT−, or Unreliable-only—**pre-register**) | ≥ 0.25 Unreliable **or** ≥ 0.40 (Unrel∪Unc) |
| Accuracy alone | **Ignored for pass/fail** |

**Hard fail:** ≥95% one class (esp. Reliable) regardless of accuracy.

### Stage 2 — Canonical A1-1000 — MEDIUM

Only Stage 1 passers. Report Spec, BalAcc, full confusion, decision histogram **alongside** Acc/F1.

### Stage 3 — Broader ablations / 5747 — HIGH

Only models with Stage 2 non-collapse + scientific need.

---

## 11. Recommended Models

### Top 5 (for *this* verification problem)

| Rank | Model | WHY TRY | WHY NOT | Scientific value | Impl. risk | Compute | Expected failure |
|------|-------|---------|---------|------------------|------------|---------|------------------|
| 1 | **InternVL3-8B-Instruct** | Grounding docs; different VL stack; size fits L40S | `trust_remote_code`; LLM still Qwen-ish | High cross-recipe comparison | Med–High | MEDIUM | Collapse or integration friction |
| 2 | **Qwen3-VL-8B-Instruct** | Grounding successor; stub config exists | Low independence | Within-family progress | Low–Med | MEDIUM | Still collapses / overfits lineage |
| 3 | **Phi-4-multimodal** | Independent; cheap | SigLIP-like vision; custom code | Diversity + cost | Med–High | LOW–MED | Same affirmative collapse |
| 4 | **Molmo-7B** or **Kimi-VL-A3B** | Independent open science / grounding | Tooling maturity | Architecture diversity | Med | MEDIUM | Load/API friction; collapse |
| 5 | **InternVL3-14B** | Scale-up if 8B works | Cost; premature if 8B fails | Dose–response | Med–High | HIGH | OOM / slow; same fail as 8B |

### Named slots

| Slot | Choice |
|------|--------|
| **BEST NEXT MODEL** | InternVL3-8B-Instruct |
| **BEST INDEPENDENT COMPARISON MODEL** | Molmo-7B **or** Phi-4-multimodal (pick after Stage 0 load test) |
| **BEST LOW-COST MODEL** | Phi-4-multimodal-instruct |
| **BEST QWEN-FAMILY SUCCESSOR** | Qwen3-VL-8B-Instruct |

---

## 12. Recommended Experiment Sequence

1. **Stage 0+1 InternVL3-8B** on balanced 100 — LOW–MEDIUM  
2. **Stage 0+1 Phi-4-multimodal** (or Molmo) — LOW–MEDIUM  
3. **Stage 0+1 Qwen3-VL-8B** — LOW–MEDIUM  
4. **Stage 2 A1-1000** only for Stage 1 passers — MEDIUM  
5. Resume **Qwen2.5 A5@5747** as primary matrix completion (orthogonal track) — HIGH  
6. **Do not** run LLaVA/Gemma A2–A5 or 5747 until a non-collapse fix is shown — avoid HIGH wasted compute  

Relative compute only: LOW / MEDIUM / HIGH (no invented GPU-hours).

---

## 13. Proposed Research Questions

1. **Why do some general-purpose VLMs exhibit majority-class collapse on imbalanced detection verification, and is the failure perceptual, preference/decoding, or both?**  
   (Motivated by LLaVA/Gemma 100% Reliable vs Qwen’s three-way outputs.)

2. **Does native multimodal training with explicit visual grounding/localization objectives improve zero-shot verification of highlighted detections relative to LLM+projector VLMs with similar language backbones?**  
   (Motivated by Qwen2.5-VL vs LLaVA-OV’s Qwen2 LLM.)

3. **Does a balanced GT+/GT− diagnostic set expose verifier failure that accuracy on a 92.8%-positive slice conceals?**  
   (Motivated by Acc=0.928 with Spec=0.)

4. **When models emit fluent morphology language while labeling all proposals Reliable, are rationales image-dependent or prompt-echo templates?**  
   (Motivated by LLaVA’s 45 unique raw texts and repeated phrases on GT-negatives.)

5. **Can overlay/dimming cues be ablated to test whether collapsed models attend to the intended region?**  
   (A1-specific; not yet run.)

---

## 14. Claims We CAN Make

- A1-1000 inputs and GT protocol are shared across Qwen/LLaVA/Gemma.
- LLaVA and Gemma predictions are **100% Reliable**; accuracy equals the always-Reliable baseline; specificity is 0.
- Qwen produces non-trivial label diversity and non-zero TN/FN structure.
- Parser failures do not explain LLaVA/Gemma Reliable rates (0 parse errors).
- LLaVA uses Qwen2 **LLM** + SigLIP; it is **not** the same system as Qwen2.5-VL.
- On disagreement samples, LLaVA/Gemma often assert palm morphology where Qwen reports ambiguity or absence—**behaviorally** consistent with collapse, not careful rejection.
- Official accuracy on this slice is a **poor** sole success metric.

---

## 15. Claims We CANNOT Yet Make

- That vision-encoder difference **causes** collapse (not ablated).
- That LLaVA/Gemma “do not see” the image (they may see it and still prefer Reliable).
- That any untested model will avoid collapse.
- That Qwen “understands palms” in a human sense—only that it discriminates under this protocol.
- That changing the word “Reliable” or label order would fix LLaVA/Gemma (untested; prior debug scripts suggest label-order sensitivity is worth probing).
- Causal ranking of InternVL3 vs Qwen3 before Stage 1.

---

## 16. Sources / References

### Internal

- `docs/EXPERIMENT_STATUS_CANONICAL.md`
- `outputs/analysis/a1_1000_cross_model_audit/`
- `src/preprocessing/verification_overlay.py`
- `src/lvm/{qwen,llava,gemma}_verifier.py`
- `configs/models/{qwen2_5_vl,llava,gemma,qwen3_vl}.yaml`
- Cluster `config.json` / preprocessor configs under `/deac/csc/yangGrp/luoz23/models/`

### External

- Bai et al., **Qwen2.5-VL Technical Report**, arXiv:2502.13923  
- Li et al., **LLaVA-OneVision**, arXiv:2408.03326  
- Google **Gemma 3** model card / HF `google/gemma-3-12b-it`  
- HF: `Qwen/Qwen3-VL-8B-Instruct`  
- HF: `OpenGVLab/InternVL3-8B-Instruct`, `InternVL3-14B`, `InternVL3_5-8B-HF`  
- HF: `microsoft/Phi-4-multimodal-instruct`; Phi-4-Mini tech report arXiv:2503.01743  
- MiniCPM-V 2.6 docs (OpenBMB)  
- Kimi-VL tech report arXiv:2504.07491  
- Molmo / PixMo (CVPR 2025)

---

## End card

**TOP RECOMMENDATION:** InternVL3-8B-Instruct — Stage 0+1 balanced gate first  

**SECOND:** Qwen3-VL-8B-Instruct — Qwen-family successor (report separately from “independent” comparisons)  

**THIRD:** Phi-4-multimodal-instruct — low-cost independent screen  

**DO NOT RUN:** LLaVA/Gemma full A2–A5 or 5747; any new model’s A1-1000/full ablations before Stage 1 pass; treating Acc/F1 alone as qualification  

**NEXT EXPERIMENT:** Build a **fixed 50/50 diagnostic subset** from existing A1-1000 GT labels and run **Stage 0+1 on InternVL3-8B** (after download/integration)—still **before** any full A1-1000 for new models. Parallel track: finish Qwen A5@5747 when GPU time allows.
