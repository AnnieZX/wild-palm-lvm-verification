# Qwen2.5-VL Full A1–A5 Results (@5747)

**Status:** `QWEN_FULL_A1_A5_COMPLETE`  
**Model:** Qwen2.5-VL-7B-Instruct  
**Checkpoint:** `/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct`  
**Experiment ID:** `20260708_0020`  
**Dataset size:** 5747 YOLO detections  

This document is the **canonical production record** for the completed Qwen full-dataset ablation (A1–A5). All metric values below were **independently recovered** from the experiment artifacts and recomputed with `scripts/compute_verification_metrics.py` / the same formulas. Numbers were **not** copied from chat prompts.

---

## 1. Canonical roots

| Role | Path |
|------|------|
| Verification (production) | `outputs/verification/qwen/20260708_0020/` |
| Evaluation (production) | `outputs/evaluation/qwen/20260708_0020/` |
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
| GT positive (`matched_gt`) | **4685** |
| GT negative | **1062** |
| Parse failures / empty decisions | **0** |
| Stored `*_metrics.json` vs recomputed TP/FP/TN/FN/P/R/F1/Acc/R/U/Ur | **match** |

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
| Recall (sensitivity) | TP / (TP+FN) | Binary subset \(B\) (not all 4685 GT+) |
| Specificity | TN / (TN+FP) | Binary subset \(B\) (not all 1062 GT−) |
| F1 | \(2PR/(P+R)\) | From binary Precision/Recall |
| Accuracy | (TP+TN) / (TP+FP+TN+FN) | Binary \(N = \|B\|\) |
| Balanced accuracy | (Recall + Specificity) / 2 | Binary subset |
| Reliable / Uncertain / Unreliable % | count / **5747** | **Full** dataset |
| Uncertain rate | Uncertain / **5747** | **Full** dataset |
| Coverage (selective) | (Reliable + Unreliable) / **5747** = 1 − Uncertain rate | **Full** dataset |

**Important:** binary Precision/Recall/F1/Accuracy/Specificity/Balanced accuracy **exclude Uncertain**. Their denominators are **not** 5747. Report Uncertain rate / Coverage separately and do not mix the two scopes.

Specificity and balanced accuracy are **not** written into `*_metrics.json` by the metrics script; they are derived here from the same TP/FP/TN/FN the script stores.

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

## 5. Final metrics table (A1–A5 @5747)

Binary metrics: Uncertain excluded. Uncertain rate & Coverage: over full 5747.

| Ablation | Reliable | Uncertain | Unreliable | Precision | Recall | Specificity | F1 | Accuracy | Balanced Accuracy | Uncertain Rate | Coverage |
|----------|----------|-----------|------------|-----------|--------|-------------|-----|----------|-------------------|----------------|----------|
| A1 | 3593 | 1775 | 379 | 0.8945 | 0.9381 | 0.3059 | 0.9158 | 0.8512 | 0.6220 | 0.3089 | 0.6911 |
| A2 | 4268 | 1344 | 135 | 0.8796 | 0.9804 | 0.1045 | 0.9273 | 0.8662 | 0.5425 | 0.2339 | 0.7661 |
| A3 | 4416 | 1313 | 18 | 0.8755 | 0.9979 | 0.0179 | 0.9327 | 0.8742 | 0.5079 | 0.2285 | 0.7715 |
| A4 | 3570 | 1557 | 620 | 0.9067 | 0.8984 | 0.4327 | 0.9026 | 0.8332 | 0.6656 | 0.2709 | 0.7291 |
| A5 | 3065 | 455 | 2227 | 0.9106 | 0.6470 | 0.7198 | 0.7565 | 0.6604 | 0.6834 | 0.0792 | 0.9208 |

### Confusion table (binary subset)

| Ablation | TP | FP | TN | FN | Binary N | Excluded Uncertain | GT+ Uncertain | GT− Uncertain |
|----------|----|----|----|----|----------|--------------------|---------------|---------------|
| A1 | 3214 | 379 | 167 | 212 | 3972 | 1775 | 1259 | 516 |
| A2 | 3754 | 514 | 60 | 75 | 4403 | 1344 | 856 | 488 |
| A3 | 3866 | 550 | 10 | 8 | 4434 | 1313 | 811 | 502 |
| A4 | 3237 | 333 | 254 | 366 | 4190 | 1557 | 1082 | 475 |
| A5 | 2791 | 274 | 704 | 1523 | 5292 | 455 | 371 | 84 |

---

## 6. Key findings (from recovered numbers only)

| Question | Answer (@5747) |
|----------|----------------|
| Highest F1 | **A3** (0.9327) |
| Highest recall | **A3** (0.9979) |
| Highest specificity | **A5** (0.7198) |
| Highest balanced accuracy | **A5** (0.6834) |
| Highest coverage | **A5** (0.9208); lowest Uncertain rate (0.0792) |

**A5 vs A1–A4:** A5 is more conservative (fewest Reliable, most Unreliable), rejects negatives far better (highest Spec / TN), but at a clear recall and F1 cost. It also abstains least often (highest coverage).

**Adding confidence / geometry (A1→A2→A3):** F1 and recall **rise**, but specificity **collapses** (0.31 → 0.10 → 0.02). Improvement is **not** monotonic for verification quality if negative discrimination matters. A3’s near-perfect recall with near-zero specificity is a warning that high F1 can coexist with almost no false-positive rejection.

**Why F1 alone misleads a second-stage FP verifier:** The set is heavily GT-positive (4685/5747 ≈ 81.5%). Policies that over-call Reliable inflate TP and F1 while failing to reject unmatched detections. Specificity and balanced accuracy expose that failure mode (especially A2/A3).

### A5 @1000 vs A5 @5747

Recovered from canonical Qwen experiment `20260706_2214` (1000 verification JSONs `sample_000001`…`sample_001000`; evaluation rows restricted to those IDs):

| Scale | Precision | Recall | F1 | Specificity | Balanced Acc |
|-------|-----------|--------|-----|-------------|--------------|
| A5 @1000 | 0.9825 | 0.6631 | 0.7918 | 0.8462 | 0.7546 |
| A5 @5747 | 0.9106 | 0.6470 | 0.7565 | 0.7198 | 0.6834 |

**Qualitative behavior reproduces:** high precision, lower recall, strong specificity relative to A1–A4. Absolute Spec/BA/F1 are somewhat milder on the full set than on the 1000-slice.

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

- `outputs/evaluation/qwen/20260708_0020/A{1–5}/A{1–5}_evaluation.csv`
- `outputs/evaluation/qwen/20260708_0020/A{1–5}/A{1–5}_metrics.json`
- `outputs/evaluation/qwen/20260708_0020/A{1–5}/summary.csv`

### Code (frozen protocol references)

- `src/prompts/ablation_verification_prompts.py`
- `scripts/evaluate_verification_against_groundtruth.py`
- `scripts/compute_verification_metrics.py`
- `src/lvm/qwen_verifier.py` (decoding: `do_sample=False`, `max_new_tokens=512`)

### Related primary @1000 ablation (separate experiment)

- `outputs/verification/qwen/20260706_2214/` (A1–A5 @1000)
- `outputs/evaluation/qwen/20260706_2214/`

---

## 9. Freeze statement

**`QWEN_FULL_A1_A5_COMPLETE`**

Treat `outputs/verification/qwen/20260708_0020/` and `outputs/evaluation/qwen/20260708_0020/` as the production full-dataset Qwen A1–A5 results. Do not archive/delete other trees in this document’s scope without a separate explicit decision.
