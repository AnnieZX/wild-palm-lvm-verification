# Qwen2.5-VL Full A1–A5 Results (@5747)

**Status:** `QWEN_FULL_A1_A5_COMPLETE`  
**Model:** Qwen2.5-VL-7B-Instruct  
**Checkpoint:** `/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct`  
**Experiment ID:** `20260708_0020`  
**Dataset size:** 5747 YOLO detections  

This document is the **production run record** for the completed Qwen full-dataset ablation (A1–A5): canonical paths, integrity checks, evaluation semantics and Qwen-specific observations. **Metric tables live only in [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md) §5 (full scale) and §7 (@1000)**, scored under Evaluation Protocol v2. Numbers quoted below are Protocol v2 unless labelled otherwise.

---

## 1. Canonical roots

| Role | Path |
|------|------|
| Verification (production) | `outputs/verification/qwen/20260708_0020/` |
| Evaluation (Protocol v2, current) | `outputs/evaluation_protocol_v2/qwen/20260708_0020/` |
| Evaluation (Protocol v1, frozen) | `outputs/evaluation/qwen/20260708_0020/` |
| Ablation inputs | `outputs/verification_ablation_5747/` |
| GT matching | LabelMe GT vs YOLO bbox, **greedy one-to-one**, **IoU ≥ 0.5** (`scripts/evaluate_verification_against_groundtruth.py`, `IOU_THRESHOLD = 0.5`) |

### NON-CANONICAL — DO NOT USE

```
outputs/verification/qwen2_5_vl/20260708_0020/A5/
```

This tree holds outputs from cancelled wrong-path job **8318954** (resume path bug). It must **not** be merged into production and must **not** be used for metrics.

---

## 2. Integrity (independently verified)

For each of A1–A5 under the canonical roots:

| Check | Result |
|-------|--------|
| Verification JSON count | **5747** |
| Unique sample IDs | **5747** |
| ID range | **1 … 5747** contiguous |
| Missing IDs | **0** |
| Duplicate IDs | **0** |
| Evaluation CSV rows | **5747** |
| GT positive / negative (`matched_gt`, Protocol v2) | **5109 / 638** (Protocol v1: 4685 / 1062) |
| Parse failures / empty decisions | **0** |
| Stored `*_metrics.json` vs recomputed TP/FP/TN/FN/P/R/F1/Acc/R/U/Ur | **match** (v1 audit 2026-09-27; v2 invariants checked by `scripts/rescore_protocol_v2.py`) |

A5 completion job: **8318977** (exit 0), writing into the legacy production tree above.

---

## 3. Evaluation semantics (from code)

Implementation: `scripts/compute_verification_metrics.py` → `compute_metrics()`.

### Decision → binary role

| Label | Role in binary metrics |
|-------|-------------------------|
| **Reliable** | predicted **positive** |
| **Unreliable** | predicted **negative** |
| **Uncertain** | **excluded** from TP/FP/TN/FN and from Precision / Recall / F1 / Accuracy |

GT label for each detection: `matched_gt` (True if greedy IoU ≥ 0.5 match exists).

### Confusion (binary subset only)

Let \(B\) = rows with label ∈ {Reliable, Unreliable}.

- **TP** = GT+ ∧ Reliable  
- **FP** = GT− ∧ Reliable  
- **FN** = GT+ ∧ Unreliable (Uncertain on GT+ is **not** FN)  
- **TN** = GT− ∧ Unreliable (Uncertain on GT− is **not** TN)  

### Formulas and denominators

| Metric | Formula | Denominator / scope |
|--------|---------|---------------------|
| Precision | TP / (TP+FP) | Binary subset \(B\) |
| Recall (sensitivity) | TP / (TP+FN) | Binary subset \(B\) (not all 5109 GT+) |
| Specificity | TN / (TN+FP) | Binary subset \(B\) (not all 638 GT−) |
| F1 | \(2PR/(P+R)\) | From binary Precision/Recall |
| Accuracy | (TP+TN) / (TP+FP+TN+FN) | Binary \(N = \|B\|\) |
| Balanced accuracy | (Recall + Specificity) / 2 | Binary subset |
| Reliable / Uncertain / Unreliable % | count / **5747** | **Full** dataset |
| Uncertain rate | Uncertain / **5747** | **Full** dataset |
| Coverage (selective) | (Reliable + Unreliable) / **5747** = 1 − Uncertain rate | **Full** dataset |

**Important:** binary Precision/Recall/F1/Accuracy/Specificity/Balanced accuracy **exclude Uncertain**. Their denominators are **not** 5747. Report Uncertain rate / Coverage separately and do not mix the two scopes.

Under Protocol v2, `specificity` and `balanced_accuracy` are written into every `*_metrics.json` by the metrics script. Protocol v1 metrics JSONs do not contain them; derive them from the stored TP/FP/TN/FN.

---

## 4. Ablation definitions (from production prompt code)

Source: `src/prompts/ablation_verification_prompts.py` (`build_ablation_verification_prompt`).

Shared across A1–A5: same role text, palm morphology cues, decision definitions (Reliable / Uncertain / Unreliable), JSON-only response schema, deterministic decoding (`do_sample=False`, `max_new_tokens=512`).

| Code | Condition folder | Visual input | Metadata in prompt |
|------|------------------|--------------|--------------------|
| **A1** | `A1_overlay_only` | Overlay patch (dimmed background + green bbox) | None; instructed **not** to use confidence or geometry |
| **A2** | `A2_overlay_confidence` | Same overlay | YOLO **confidence** only (auxiliary) |
| **A3** | `A3_overlay_confidence_geometry` | Same overlay | Confidence + bbox width/height/area + aspect ratio |
| **A4** | `A4_overlay_crop_confidence` | **Two-panel:** overlay + enlarged crop | YOLO confidence (auxiliary) |
| **A5** | `A5_crop_only` | **Crop only** (no surround context) | YOLO confidence (auxiliary) |

---

## 5. Metrics (Protocol v2)

Full A1–A5 metrics, confusion counts and failure counts: [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md) §5 (row group "Qwen2.5-VL-7B"). The @1000 experiment `20260706_2214` is in §7 of the same file.

Uncertain answers split by GT class (Protocol v2; not reported elsewhere). Uncertain on GT+ is neither TP nor FN; Uncertain on GT− is neither FP nor TN.

| Ablation | Binary N | Uncertain | GT+ Uncertain | GT− Uncertain |
|----------|---------:|----------:|--------------:|--------------:|
| A1 | 3972 | 1775 | 1421 | 354 |
| A2 | 4403 | 1344 | 988 | 356 |
| A3 | 4434 | 1313 | 940 | 373 |
| A4 | 4190 | 1557 | 1221 | 336 |
| A5 | 5292 | 455 | 400 | 55 |

---

## 6. Key findings (Protocol v2)

| Question | Answer (@5747) |
|----------|----------------|
| Highest F1 | **A3** (0.9694) |
| Highest recall | **A3** (0.9981) |
| Highest specificity | **A5** (0.9005) |
| Highest balanced accuracy | **A5** (0.7695), narrowly ahead of A4 (0.7628) |
| Highest coverage | **A5** (0.9208); lowest Uncertain rate (0.0792) |

**A5 vs A1–A4:** A5 is more conservative (fewest Reliable, most Unreliable), rejects negatives far better (highest Spec / TN), but at a clear recall and F1 cost. It also abstains least often (highest coverage).

**Adding confidence / geometry (A1→A2→A3):** F1 and recall **rise**, but specificity **collapses** (0.42 → 0.18 → 0.04). Improvement is **not** monotonic for verification quality if negative discrimination matters. A3's near-perfect recall with near-zero specificity is a warning that high F1 can coexist with almost no false-positive rejection.

**Why F1 alone misleads a second-stage FP verifier:** The set is heavily GT-positive (5109/5747 ≈ 88.9%). Policies that over-call Reliable inflate TP and F1 while failing to reject unmatched detections. Specificity and balanced accuracy expose that failure mode (especially A2/A3).

### A5 @1000 vs A5 @5747

The @1000 slice (`20260706_2214`, `sample_000001`…`sample_001000`; unaffected by the v2 correction) has A5 Spec 0.8462, BalAcc 0.7546, F1 0.7918; the full set under Protocol v2 has A5 Spec 0.9005, BalAcc 0.7695, F1 0.7736.

**Qualitative behavior reproduces:** high precision, lower recall, strong specificity relative to A1–A4. Under Protocol v2, full-set specificity and balanced accuracy are slightly **higher** than on the 1000-slice and F1 is slightly lower. (Protocol v1 had reported the full-set values as milder; that comparison was affected by the case-sensitive GT label defect.)

---

## 7. Limitations

1. **Uncertain exclusion.** Binary metrics use denominators over Reliable∪Unreliable only. Uncertain rate and Coverage must be reported separately; do not treat binary Accuracy as “accuracy over 5747.”

2. **Frozen decision protocol.** These results characterize Qwen under the canonical Reliable / Uncertain / Unreliable protocol. They do **not** establish prompt- or verbalizer-independent robustness.

3. **Label / token / order sensitivity.** A separate diagnostic found decision-interface sensitivity for Qwen (order permutation especially). Canonical artifacts:  
   `outputs/diagnostics/model_qualification/qwen_label_robustness_20/`  
   (see `EVALUATION_SUMMARY.txt`, `EVALUATION_REPORT.json`). Interpretation there: **SEVERELY_SENSITIVE** (n=20 controlled probe; not a benchmark claim).

4. **GT vs prompt semantics.** GT correctness is **greedy IoU ≥ 0.5** match to LabelMe palms. The VLM prompt asks whether the highlighted box contains a **valid wild palm** (morphology). Alignment between geometric match and the prompt’s validity notion remains an open **methodological audit** item and is **not** resolved by these numbers.

---

## 8. Canonical artifact checklist

### Verification

- `outputs/verification/qwen/20260708_0020/A1/` … `A5/` (each: 5747 × `sample_*.json`)

### Evaluation

- `outputs/evaluation_protocol_v2/qwen/20260708_0020/A{1–5}/A{1–5}_evaluation.csv` (Protocol v2, current)
- `outputs/evaluation_protocol_v2/qwen/20260708_0020/A{1–5}/A{1–5}_metrics.json`
- `outputs/evaluation_protocol_v2/qwen/20260708_0020/A{1–5}/summary.csv`
- `outputs/evaluation_protocol_v2/qwen/20260708_0020/PROTOCOL.json`
- `outputs/evaluation/qwen/20260708_0020/…` (same files, Protocol v1, frozen provenance)

### Code (frozen protocol references)

- `src/prompts/ablation_verification_prompts.py`
- `scripts/evaluate_verification_against_groundtruth.py`
- `scripts/compute_verification_metrics.py`
- `src/lvm/qwen_verifier.py` (decoding: `do_sample=False`, `max_new_tokens=512`)

### Related primary @1000 ablation (separate experiment)

- `outputs/verification/qwen/20260706_2214/` (A1–A5 @1000)
- `outputs/evaluation_protocol_v2/qwen/20260706_2214/` (v1 twin: `outputs/evaluation/qwen/20260706_2214/`)

---

## 9. Freeze statement

**`QWEN_FULL_A1_A5_COMPLETE`**

Treat `outputs/verification/qwen/20260708_0020/` (predictions) and `outputs/evaluation_protocol_v2/qwen/20260708_0020/` (Protocol v2 evaluation) as the production full-dataset Qwen A1–A5 results; `outputs/evaluation/qwen/20260708_0020/` is the frozen Protocol v1 evaluation. Do not archive/delete other trees in this document’s scope without a separate explicit decision.
