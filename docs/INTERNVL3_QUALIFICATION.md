# InternVL3 Qualification Report

**Status:** Stage 1 complete — **FAIL_TECHNICAL** (not authorized for A1-1000)  
**Date:** 2026-09-09 (DEAC)  
**Experiment ID:** `20260909_internvl3_qual`

---

## 1. Exact model

- **HF repo:** `OpenGVLab/InternVL3-8B-Instruct`
- **Local path:** `/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct`
- **Registry key:** `internvl3` (alias `internvl`)

## 2. HF revision

- **Revision/commit:** `ddb3a169d5582e5c76e0809a128e55ab63686ada`
- Recorded in `DOWNLOAD_META.txt` (download Slurm job `8303105`).

## 3. Model path

`/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct` (~15.9 GB; 4 safetensors shards present).

## 4. Integration details

Official HF card (not guessed from other models):

| Item | Finding |
|------|---------|
| transformers | `>=4.37.2` (cluster 4.57.6) |
| trust_remote_code | **Required** (`True`) |
| APIs | `AutoModel` + `AutoTokenizer` (`use_fast=False`); `model.chat(...)` |
| Image preprocess | Official dynamic 448×448 tiling + ImageNet normalize; `max_num`≤12 + thumbnail |
| Chat | Required `<image>` marker; frozen prompt body unchanged |
| dtype | BF16 |
| flash-attn | Optional; falls back to eager |
| GPU used | Stage 0/1 on `gpu_small` / `V100_32:1` (Slurm) |

Additive code only; frozen eval/parser/GT untouched.

Generation: `do_sample=False`, `max_new_tokens=512`, `dtype=bfloat16`.

## 5. Prompt parity

CPU check `scripts/check_internvl_prompt_parity.py`: semantic verification instruction identical to Qwen after stripping required InternVL `<image>\n` marker. Same canonical A1 prompt/image paths.

## 6. Stage 0 manifest

`outputs/diagnostics/model_qualification/stage0_A1_10/`  
Parent: `balanced_A1_100/` seed `20260908`.  
4 clear GT+ / 4 clear GT− / 2 near-threshold (by `max_iou`).

## 7. Stage 0 result

- Slurm job **8303119**
- Outputs: `outputs/verification/internvl3/20260909_internvl3_qual/stage0/`
- **10/10** completed, status `ok`, **0** parse/inference failures
- Predictions: Reliable **4**, Uncertain **6**, Unreliable **0**
- Clear GT+ → Reliable; clear GT− → Uncertain
- Unique raw texts: 9/10
- **Stage 0: PASS** (technical gate)

## 8. Balanced-100 manifest

`outputs/diagnostics/model_qualification/balanced_A1_100/`  
Seed `20260908`; **50** GT+ / **50** GT− from canonical A1-1000; path references only (no regenerated prompts/overlays).

## 9–11. Stage 1 result / decision distribution / confusion

- Slurm job **8303143** (`gpu_small`, `V100_32:1`)
- Outputs: `outputs/verification/internvl3/20260909_internvl3_qual/balanced100/`
- Expected **100**; completed **100**; missing **0**
- Index status: `ok` **87**, `parse_error` **13**, inference failures **0**

### Predictions (n=100)

| Decision | Count | % |
|----------|------:|--:|
| Reliable | 56 | 56% |
| Uncertain | 31 | 31% |
| Unreliable | 0 | 0% |
| *(empty / parse fail)* | 13 | 13% |

### Official binary confusion (Reliable=pos, Unreliable=neg, Uncertain excluded)

Scored n = **56** (all Reliable; zero Unreliable).

| | |
|--|--:|
| TP | 35 |
| FP | 21 |
| TN | 0 |
| FN | 0 |

### Metrics (official binary)

| Metric | Value |
|--------|------:|
| Precision | 0.6250 |
| Recall | 1.0000 |
| F1 | 0.7692 |
| Accuracy | 0.6250 |
| **Specificity** | **0.0000** |
| **Balanced Accuracy** | **0.5000** |

### Diagnostic GT-negative rejection (does **not** replace official binary)

| GT− → | Count |
|-------|------:|
| Reliable | 21 |
| Uncertain | 20 |
| Unreliable | 0 |
| parse fail (empty) | 9 |
| Rejection rate (Uncertain∨Unreliable) / 50 | **0.4000** |

## 12. Specificity

**0.0000** (no Unreliable predictions → TN=0 among scored negatives).

## 13. Balanced Accuracy

**0.5000**

## 14. Failure / collapse analysis

- **Not** single-class hard collapse (max class Reliable = 56% < 95%).
- Raw-response diversity: **70** unique texts / 100; visual_reasoning **66** unique; top raw repeated 13× (malformed JSON stub).
- Rationales often share palm-template phrasing but vary by sample; Uncertain cases cite weak crown/frond evidence.
- **Parser:** 13/100 failures — incomplete JSON of form `{"decision": "Reliable",}` (trailing comma / missing fields). Parser success **87% < 95%** gate.
- **Scientific gates also fail:** Specificity 0.00 < 0.20; Balanced Accuracy 0.50 < 0.55; never emits Unreliable.

## 15. Verdict

**FAIL_TECHNICAL**

Primary: parser success 87% < pre-registered 95%.  
Also fails Specificity / Balanced Accuracy minima; zero Unreliable class usage.

## 16. A1-1000 authorization

**QUALIFIED FOR A1-1000: NO**

Do not run A1-1000 for InternVL3 until parser reliability and negative-class (Unreliable) behavior are addressed and Stage 1 is re-run on the **same** fixed balanced-100 (no prompt rewrite).

---

## 17. Stage 1 failure diagnosis (2026-09-09)

Shared parser **not** modified. No new inference. Diagnosis from frozen Stage 1 JSON only.

### 17.1 All 13 parse failures

Every failure is **byte-identical**:

```text
```json
{
  "decision": "Reliable",
}
```
```

| sample_id | GT | raw (abbrev) | visual_reasoning | parser error | recoverable intended |
|-----------|-----|--------------|------------------|--------------|----------------------|
| sample_000071 | positive | stub above | *(empty)* | `Expecting property name…` (trailing comma) | Yes → **Reliable** |
| sample_000088 | negative | stub | empty | same | Yes → Reliable |
| sample_000151 | negative | stub | empty | same | Yes → Reliable |
| sample_000205 | negative | stub | empty | same | Yes → Reliable |
| sample_000207 | negative | stub | empty | same | Yes → Reliable |
| sample_000215 | negative | stub | empty | same | Yes → Reliable |
| sample_000220 | negative | stub | empty | same | Yes → Reliable |
| sample_000263 | negative | stub | empty | same | Yes → Reliable |
| sample_000274 | negative | stub | empty | same | Yes → Reliable |
| sample_000368 | positive | stub | empty | same | Yes → Reliable |
| sample_000379 | positive | stub | empty | same | Yes → Reliable |
| sample_000626 | positive | stub | empty | same | Yes → Reliable |
| sample_000868 | negative | stub | empty | same | Yes → Reliable |

**Malformed categories (counts; co-occurring):**

| Category | Count |
|----------|------:|
| trailing_comma | **13** |
| incomplete_object (decision only; no reasoning fields) | **13** |
| decision_only_stub | **13** |
| truncated_generation | **0** |
| invalid quote | **0** |
| extra prose | **0** |
| duplicated keys | **0** |
| other | **0** |

Notes: raw length **41** vs successful mean ~**560**. Runtimes ~3.6s. Not `max_new_tokens=512` truncation — generation ends after a short invalid stub with closed fences.

### 17.2 Where malformed JSON is introduced

Trace: frozen prompt file → InternVL `<image>\n` + `model.chat` (official remote-code; sets `eos_token_id` from conversation `sep`) → decoded assistant text → shared `parse_verification_response` (strict JSON).

| Hypothesis | Status |
|------------|--------|
| Model emits non-compliant JSON (decision-only + illegal trailing comma) | **SUPPORTED** |
| Shared parser correctly rejects invalid JSON (no silent Reliable default) | **SUPPORTED** |
| Truncation via `max_new_tokens` | **Not supported** (41-char complete stubs) |
| Stop/EOS cutting mid-object | **Not supported** (closed `}` and ```` ``` ````) |
| Decoder cleanup stripping fields | **UNKNOWN** (no evidence of post-decode mutation in adapter) |
| Prompt wording causes stubs | **PLAUSIBLE** (model sometimes short-circuits schema) |
| Chat template / system message interaction | **PLAUSIBLE** |
| Official API misuse | **Not supported** for this symptom (`chat` + bf16 + `do_sample=False` matches card) |

### 17.3 Zero `Unreliable` (GT-negative n=50)

| Decision | Count |
|----------|------:|
| Reliable | 21 |
| Uncertain | 20 |
| Unreliable | **0** |
| parse-fail (empty) | 9 |

Word **`Unreliable` never appears in any of the 100 raw responses.**

Evidence-backed interpretations:

| Code | Finding |
|------|---------|
| **A** | **SUPPORTED** for GT− → Uncertain (20/20): all use “not clear / lacks clear / not distinctly” palm morphology language; **0** use “consistent with … wild palm”. |
| **B** | **SUPPORTED** for GT− → Reliable (21): **15/21** use “consistent with the characteristics of a wild palm”; **0** use not-clear language — model asserts palm presence (FP belief), not rejection. |
| **C** | **SUPPORTED**: even when evidence is denial-like (Uncertain), label chosen is Uncertain, never Unreliable; Unreliable absent corpus-wide. |
| **D** | **UNKNOWN** without a rename probe (possible aversion to the token “Unreliable”). |
| **E** | **Not supported**: 70 unique raws; Uncertain vs Reliable rationales differ systematically. |
| **F** | **PLAUSIBLE** as secondary description of A+C (prefers Uncertain over hard REJECT). |

### 17.4 Recommended label-semantics probe (NOT RUN)

**One probe only:** on a fixed tiny subset of the **same** balanced-100 images/GT (recommend **20 GT-negatives**: 10 that were Reliable + 10 that were Uncertain under Stage 1), keep visual task identical, change **only** decision vocabulary / JSON enum to:

- `ACCEPT` ← Reliable  
- `REVIEW` ← Uncertain  
- `REJECT` ← Unreliable  

Same decision definitions remapped to those names; no added visual hints; no shared-parser loosening (map labels only in a **probe-local** eval script).

**Readout:** If `REJECT` appears on the prior-Uncertain GT− set → label-token/`Unreliable` semantics likely blocked hard negatives. If `REJECT` still never appears → model decision policy (maps non-palm evidence to REVIEW only, or still affirms ACCEPT) → unsuitable under this instruction family without larger redesign.

### 17.5 Failure classification

**C. MIXED_TECHNICAL_AND_MODEL_FAILURE**

- Technical: 13% identical invalid JSON stubs fail the ≥95% parser gate.  
- Model: among 87 valid parses, **zero** Unreliable → Spec=0, BalAcc=0.5; GT− split between palm-affirming Reliable and denial→Uncertain.

Fixing JSON alone would mostly add more Reliable (all 13 stubs declare Reliable), which does **not** repair Specificity.

### 17.6 Next action (exactly one)

**Superseded by §18 probe result.** Label rename is not a viable fix (20/20 REJECT collapse). InternVL remains not qualified; do not run A1-1000.

---

## 18. Label-Semantics Probe: ACCEPT / REVIEW / REJECT

**Status:** COMPLETE (diagnostic only). InternVL qualification **unchanged: STILL NOT QUALIFIED** (not PASS).

| Item | Value |
|------|-------|
| Job | **8303173** (`gpu_small` / `V100_32:1`) |
| Job state | **COMPLETED** (log: 20/20 `ok`; ended `Wed Sep 9 12:36:26 AM EDT 2026`) |
| Subset | `outputs/diagnostics/model_qualification/internvl_label_semantics_20/` |
| N / GT | 20 = 10 GT+ + 10 GT− (lowest `canonical_order` per class in balanced-100) |
| Label map | Reliable→ACCEPT, Uncertain→REVIEW, Unreliable→REJECT |
| Results | `…/internvl_label_semantics_20/results_20260909_internvl3_qual/` |
| Shared parser | **unchanged** |

### Completion

| Metric | Value |
|--------|------:|
| Expected | 20 |
| Completed JSON | 20 |
| Missing | 0 |
| Inference failures | 0 |
| Parse failures | **0** |
| Probe parse success | **100%** |

### Original vs renamed counts (same 20 IDs)

| | Count | % |
|--|------:|--:|
| **Original** Reliable | 10 | 50% |
| Original Uncertain | 8 | 40% |
| Original Unreliable | **0** | 0% |
| Original parse_fail | 2 | 10% |
| **Renamed** ACCEPT | **0** | 0% |
| Renamed REVIEW | **0** | 0% |
| Renamed REJECT | **20** | **100%** |
| Renamed parse_fail | **0** | 0% |

Parse success on these 20: original **18/20 (90%)** → renamed **20/20 (100%)**.  
Trailing-comma stub pattern: **did not recur** on this probe.

### GT-negative transitions (n=10)

| Transition | Count |
|------------|------:|
| Reliable → REJECT | 3 |
| Uncertain → REJECT | 6 |
| parse_fail → REJECT | 1 |
| **GT-negative REJECT** | **10 / 10** |

All 10 GT− sample_ids became REJECT:  
`sample_000019, 000025, 000036, 000074, 000077, 000088, 000093, 000129, 000130, 000140`.

### GT-positive behavior (n=10)

| ACCEPT | REVIEW | REJECT | parse_fail |
|-------:|-------:|-------:|-----------:|
| **0** | **0** | **10** | **0** |

**GT-positive ACCEPT: 0 / 10** — positive recognition destroyed under rename.

### Raw-response notes

- Unique renamed raws: **18 / 20**
- Token `REJECT` in every response; `Reliable`/`Unreliable` absent
- GT− rationales explicitly deny palm morphology (central crown / radial fronds absent)
- GT+ rationales use the **same denial template** (“does not contain a visible central crown…”) despite Stage-1 often Reliable — reasoning changed with the label set, not merely the final enum token
- Effect is **not** selective image-dependent 3-way labeling; it is **single-class REJECT collapse**

### Probe verdict

**INCONCLUSIVE**

- Not `LABEL_SEMANTICS_EFFECT_SUPPORTED`: requires GT+ mostly ACCEPT; observed **0/10** ACCEPT.  
- Not `LABEL_SEMANTICS_EFFECT_NOT_SUPPORTED` as written (zero REJECT): observed **20/20** REJECT.  
- Technically complete (no parse/infer failure), but the all-REJECT collapse prevents a reliable paired conclusion that renaming `Unreliable`→`REJECT` selectively fixes zero-Unreliable behavior.

Implication: InternVL can emit a hard-reject class token when named `REJECT`, but under this controlled rename it **collapses to always-REJECT**, so label renaming is **not** a viable production fix and does **not** clear Stage-1 qualification.
