# RQ evidence matrix

Status as of 2026-10-02. All numbers are descriptive and come from `paper/analysis/outputs/`.
Seven models have all five conditions (the *complete set*): Qwen2.5-VL-7B, Qwen3-VL-8B,
Qwen3-VL-4B, InternVL3.5-8B, GLM-4.6V-Flash, Phi-4-multimodal, Gemma-4-12B.
MiniCPM-V-4.5 and Molmo2-8B have A1 only. Qwen3-VL-2B is running and is excluded.

Vocabulary:
- "Alignment" means Protocol v2 IoU agreement with LabelMe. It is not correctness.
- "Matched/unmatched" is an IoU outcome. "Palm/non-palm/ambiguous" is the human semantic label.
- Bootstrap CIs are image-level resampling intervals. They are not significance tests.

---

## RQ1: How does the input representation (A1–A5) affect behaviour and alignment?

**Analyses needed.** Separate the two kinds of contrast:

- **Prompt/metadata effects** on an identical image: A1→A2 (adds confidence) and A2→A3 (adds geometry).
- **Image-representation effects** with identical confidence text: A2→A4 (adds crop) and A2→A5 (crop only), plus A4→A5. These also change the image-description paragraph, and A2→A4/A5 change one instruction line, so they are not image-only.
- Mixed contrasts (A1→A4/A5, A3→A4/A5) confound both kinds and should not be interpreted as either.

**Available evidence**

| Output | Path |
|---|---|
| Behaviour + alignment per model × condition | `outputs/master_results.csv`; `tables/tab_master_behavior.tex`, `tab_master_alignment.tex` |
| Sample-level 3×3 transitions | `outputs/transition_*.csv`; `figures/fig_transitions_text_only`, `fig_transitions_image_representation`, `fig_transitions_mixed_image_and_metadata`; `tables/tab_transitions_summary.tex` |
| Point deltas | `outputs/condition_deltas.csv`; `tables/tab_deltas_image_representation.tex`, `tab_deltas_text_only.tex`; `figures/fig_deltas_image_representation` |
| Paired image-level bootstrap CIs | `outputs/bootstrap_delta_cis.csv`, `bootstrap_transition_cis.csv`; `tables/tab_bootstrap_deltas.tex`; `figures/fig_bootstrap_deltas` |

**What can be said descriptively now**

- **A2→A5 (image representation).** All seven complete-set models move net toward rejection (net toward-rejection rate +0.06 for Gemma-4 to +0.55 for Qwen3-VL-8B).
  - It is the largest change for most models: 73% of Qwen3-VL-8B verdicts change, versus 9% for Gemma-4.
  - Qwen3-VL-8B's change in A5 is mostly into Uncertain (U rate 0.07→0.69). Qwen2.5-VL-7B's is mostly into Unreliable (Ur rate 0.02→0.39).
- **A1→A2 (text: add confidence).** Net toward acceptance for all seven models. Small for Phi-4 and Gemma-4 (|net| ≤ 0.005); 0.09–0.15 for the Qwen models and InternVL3.5.
- **A2→A3 (text: add geometry).** Mostly small (change rate 0.03–0.07). Exceptions: Phi-4 (0.153, entirely toward rejection) and Qwen3-VL-8B (0.110).
- **A2→A4.** The direction is model-dependent.
  - Alignment specificity rises for Qwen2.5-VL-7B (+0.46) and Phi-4 (+0.58).
  - It falls for Qwen3-VL-8B (−0.27).
  - InternVL3.5's CI includes 0.
- **Alignment specificity is based on only 638 unmatched detections, 97% of which a human judged to be palms.** Changes in it mostly reflect changes in rejecting LabelMe-unmatched palms; see RQ4.

**Must NOT yet be concluded**

- That any change is caused by the visual content alone. Every A2/A4/A5 contrast also changes prompt text.
- That a condition is "best" or "improves reliability". Alignment metrics are not semantic correctness, and A5 gains in specificity coincide with large drops in coverage or sensitivity.
- Statistical significance from non-overlapping CIs.
- Anything about A2–A5 for MiniCPM-V-4.5, Molmo2-8B or Qwen3-VL-2B.

**Missing.** A2–A5 for MiniCPM-V-4.5 and Molmo2-8B (queued). An image-only control condition, which would need a new prompt variant; this is a design decision, not just an analysis.

---

## RQ2: Are effects consistent across families and scales?

**Analyses needed.**
- Per-model deltas and transitions side by side (done).
- Within-family scale contrasts:
  - Qwen3-VL 2B/4B/8B;
  - InternVL3.5 2B/4B/8B/14B;
  - Qwen2.5-VL 3B/7B (and 32B probes).
- Cross-model agreement per condition (done).

**Available evidence**

| Output | Path |
|---|---|
| Deltas and bootstrap CIs per model | `outputs/condition_deltas.csv`, `outputs/bootstrap_delta_cis.csv`; `figures/fig_bootstrap_deltas` |
| Agreement per condition | `outputs/agreement_pairwise.csv`, `agreement_by_condition_*.csv`; `tables/tab_agreement_by_condition.tex`; `figures/fig_agreement_heatmaps` |
| Counted parameter sizes | `outputs/audit/model_inventory.csv` |

**What can be said descriptively now**

- **Consistent across all seven:** the direction of A2→A5 (toward rejection) and of A1→A2 (toward acceptance). The magnitudes differ by an order of magnitude.
- **Inconsistent:** the direction of A2→A4 on alignment specificity and balanced accuracy, which depends on the model.
- **Cross-model agreement is low in every condition.**
  - Seven-model Light's κ is 0.16–0.21; Fleiss' κ is 0.13–0.19; mean pairwise raw agreement is 0.44–0.65.
  - A5 has the lowest raw agreement (0.44); only 8.5% of detections get a unanimous verdict there.
- **High raw agreement can coexist with κ ≈ 0.1 under prevalence imbalance.** For example, Phi-4 and Gemma-4 agree on 0.87 of A2 items, but κ = 0.12 and κ_max = 0.31.
- **Qwen3-VL-4B behaves very differently from Qwen3-VL-8B.** It answers Unreliable 43–57% of the time in every condition, versus 7–21% for the 8B. This is a single within-family pair, not a scaling trend.

**Must NOT yet be concluded**
- Any scaling relationship. Only one within-family pair (Qwen3-VL 4B vs 8B) is complete, and family and size are confounded across the other models.
- That κ differences indicate better or worse models. κ is bounded by each model's marginals (see κ_max).

**Missing / blocked by running jobs**
- Qwen3-VL-2B: A1 4,257/5,747 and A2 3,009/5,747 at last check; A3–A5 pending.
- InternVL3.5 2B/4B/14B, Qwen2.5-VL-3B, MiniCPM-V-4.5 A2–A5, Molmo2-8B A2–A5: all pending.
- 32B and Ministral3: probes only.

---

## RQ3: How do models use R, U and Ur (Reliable-heavy behaviour, abstention)?

**Analyses needed.** Decision distributions per model × condition, use or non-use of U, ranges across conditions, and transitions into and out of U. No collapse threshold is defined, and none is introduced here.

**Available evidence**

| Output | Path |
|---|---|
| Decision distributions | `outputs/decision_distribution_{long,wide,ranges}.csv`; `tables/tab_decision_distribution.tex`; `figures/fig_decision_distribution_stacked`, `fig_decision_rates_by_condition` |
| Abstention-heavy and consensus cases | `outputs/case_selection/case_manifest.csv` |

**What can be said descriptively now**

- **Never use Uncertain:** Phi-4-multimodal and Gemma-4-12B, in all five conditions (0 of 28,735 verdicts each). MiniCPM-V-4.5 A1 is also 0.
- **Reliable-heavy:** Gemma-4 R share is 0.92–0.97, and MiniCPM-V-4.5 A1 is 0.97. Both are close to the always-Reliable reference (alignment accuracy 0.889).
- **Abstention-heavy:**
  - Molmo2-8B A1: U = 0.79, Ur = 0.000.
  - InternVL3.5-8B: U 0.39–0.71, Ur ≤ 0.03.
  - Qwen3-VL-8B in A5: U 0.69.
- **Unreliable-heavy:** Qwen3-VL-4B, Ur 0.44–0.57.
- **Uncertain use depends on the condition.** A5 raises the U share for Qwen3-VL-8B, InternVL3.5, GLM-4.6V and Qwen3-VL-4B. It lowers it for Qwen2.5-VL-7B (0.23→0.08).

**Must NOT yet be concluded**
- "Collapse" as a binary label. No threshold is defined in the protocol; report shares.
- That abstention is "good" or "calibrated". There is no per-item correctness reference for the U items.

**Missing.** Agreement on an operational definition of collapse, if the thesis wants one. It must be fixed before looking at more models.

---

## RQ4: What limits does IoU matching impose as a reference?

**Analyses needed.**
- Keep the IoU status separate from the semantic label.
- Composition of the unmatched set.
- Why detections are unmatched.
- VLM verdicts within each semantic class.
- Sensitivity of matched/unmatched counts to the threshold.
- The pilot, kept separate.

**Available evidence**

| Output | Path |
|---|---|
| Review audit (SHA-256 verified, complete) | `outputs/semantic/semantic_audit.json` |
| Semantic review tables | `tables/tab_semantic_review.tex`, `tab_unmatched_reasons.tex`, `tab_semantic_by_decision.tex`; `figures/fig_semantic_verdicts` |
| Semantic make-up of alignment cells | `outputs/semantic/alignment_cells_semantic_composition.csv` |
| IoU sensitivity (provenance verified, recomputed) | `outputs/iou_sensitivity/*`; `tables/tab_iou_sensitivity.tex` |
| Pilot | `outputs/semantic/pilot_by_confidence_bin.csv`; `tables/tab_pilot_by_bin.tex` |

**What can be said descriptively now**

- **The unmatched set is almost all palms.** All 638 LabelMe-unmatched detections were reviewed: 619 palm (97.0%), 19 ambiguous, 0 non-palm. This holds in the historical subsets too (balanced-100 negatives 45/5/0; A1@1,000 slice 65/7/0).
- **Why detections are unmatched**, derived from geometry:
  - 338 have no overlapping LabelMe box;
  - 275 have best IoU in (0, 0.5);
  - 25 have IoU ≥ 0.5 but lost the greedy one-to-one assignment.
- **Alignment "TN" and "FP" are not semantic outcomes.** Under Protocol v2, every Unreliable verdict on an unmatched detection counts as a "TN", yet in this cohort no such detection is a confirmed non-palm. Examples of the share of the 619 semantic palms rejected:
  - Qwen2.5-VL-7B A5: 0.82 (its alignment specificity is 0.90);
  - Phi-4 A4: 0.85;
  - Qwen3-VL-4B: 0.63–0.82 in every condition.

  Gemma-4 accepts 0.68–0.92 of them.
- **The threshold choice matters.** At IoU 0.3, 81 detections become matched (80 palm, 1 ambiguous). At 0.7, 390 matched detections become unmatched, and these have no semantic labels.
- **Pilot (separate population).** Lower-confidence detections (0.10–0.50) do include non-palms: 9 of 400, with 73 ambiguous and 318 palm. Bins are equal-sized, so these are per-bin shares, not prevalence. There are no VLM predictions for these items.

**Must NOT yet be concluded**

- That any model has a "false-positive" problem. No semantic non-palm exists in the cohort.
- Any semantic precision or specificity. Matched detections were never reviewed, so semantic errors among the 5,109 are unknown, and there are no semantic negatives to compute specificity on.
- That pilot non-palm shares transfer to the cohort. They come from a different confidence range and are equal-allocated by bin.
- That a different IoU threshold is "more correct".

**Missing.**
- A semantic audit of a random sample of matched detections, to bound errors among matched detections.
- VLM runs on a population that contains semantic non-palms, such as the pilot detections, if semantic specificity is a thesis goal.
- A second reviewer for inter-rater reliability. The current review is single-reviewer.
