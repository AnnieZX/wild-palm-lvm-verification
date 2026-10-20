# Final figure and table inventory

Generated after the 2026-10-19 analysis rebuild (`bash paper/analysis/run_all.sh`).
Every path is under `paper/analysis/` unless noted. Input decisions come from
`paper/analysis/derived/sample_decisions.csv.gz` (valid rows of audited cells) plus
read-only `outputs/` artifacts. Raw prediction JSON, Protocol-v2 evaluation CSVs,
and human-review files were not modified.

**Cohort.** Frozen YOLO detections, *N* = 5,747 on 870 images. Protocol-v2 matching:
5,109 LabelMe-matched / 638 unmatched (IoU ≥ 0.5, greedy one-to-one).

**Panels.** Thesis primary tables stay on `CORE_PANEL` (7 models). Later checkpoints
(Qwen3-VL-2B/32B, InternVL3.5-2B/4B/14B, Qwen2.5-VL-3B, MiniCPM-V-4.5, Molmo2-8B)
are in `*_extended` tables, the size ladder, and the CSVs. Do not mix panels in one
comparison without stating the cohort.

**Constructs (do not collapse).**
1. Ground-truth-free R/U/Ur behaviour.
2. Protocol-v2 LabelMe alignment (not semantic correctness).
3. Human semantic review of unmatched detections only (619 palm / 19 ambiguous / 0 non-palm).

Alignment Acc/F1 exclude Uncertain and must be read with coverage and the Always-Reliable
reference (Acc 0.889, F1 0.941). Alignment specificity is not semantic specificity.
Semantic non-palm specificity is not estimable from this cohort.

| ID | Title | RQ | Script | Input | Generated path | Cohort / denominator | Thesis | Slides |
|---|---|---|---|---|---|---|---|---|
| T1 | Decision behavior, seven-model panel | RQ1, RQ3 | `scripts/a01_master_table.py` | `derived/sample_decisions.csv.gz`; `outputs/audit/audit_cells.csv` | `tables/tab_master_behavior.tex`; source CSV `outputs/master_results.csv` | CORE_PANEL × A1–A5; *N* = 5,747 per cell; coverage = (R+Ur)/N | yes | yes |
| T2 | Decision behavior, further checkpoints | RQ1, RQ3 | `scripts/a01_master_table.py` | same | `tables/tab_master_behavior_extended.tex` | All non-CORE models with usable cells; per-cell valid *N*; † = partial | appendix | no (busy) |
| T3 | Protocol-v2 alignment, seven-model panel | RQ1, RQ4 | `scripts/a01_master_table.py` | same + Protocol-v2 metrics | `tables/tab_master_alignment.tex` | CORE_PANEL; decided rows only for Acc/F1; Always-Reliable row included | yes | yes (with coverage + baseline) |
| T4 | Protocol-v2 alignment, further checkpoints | RQ1, RQ4 | `scripts/a01_master_table.py` | same | `tables/tab_master_alignment_extended.tex` | Non-CORE usable cells; Always-Reliable row included | appendix | no |
| T5 | Decision distribution R/U/Ur % | RQ1, RQ3 | `scripts/a02_decision_distributions.py` | `derived/sample_decisions.csv.gz` | `tables/tab_decision_distribution.tex`; CSVs `outputs/decision_distribution_*.csv` | All analysed models; *N* = 5,747 unless † | yes | yes |
| F1 | Stacked R/U/Ur bars by condition | RQ1, RQ3 | `scripts/a02_decision_distributions.py` | same | `figures/fig_decision_distribution_stacked.{png,pdf}` | All analysed models; shaded A1–A3 = identical image | yes | yes |
| F2 | Per-label rate vs condition | RQ1, RQ3 | `scripts/a02_decision_distributions.py` | same | `figures/fig_decision_rates_by_condition.{png,pdf}` | Same; lines are not an ordered scale | yes | yes |
| T6 | Paired decision-change rates | RQ1 | `scripts/a03_transitions.py` | paired valid IDs in `derived/sample_decisions.csv.gz` | `tables/tab_transitions_summary.tex`; `outputs/transition_*.csv` | CORE_PANEL; *N* = 5,747 per pair | yes | yes |
| T7 | Paired decision-change rates, further checkpoints | RQ1 | `scripts/a03_transitions.py` | same | `tables/tab_transitions_summary_extended.tex` | Non-CORE models with all five conditions; paired valid *N* | appendix | no |
| F3 | Transition heatmaps, text-only | RQ1 | `scripts/a03_transitions.py` | same | `figures/fig_transitions_text_only.{png,pdf}` | All analysed models; A1↔A2↔A3, identical image | yes | yes |
| F4 | Transition heatmaps, image representation | RQ1 | `scripts/a03_transitions.py` | same | `figures/fig_transitions_image_representation.{png,pdf}` | A2/A4/A5; not image-only | yes | yes |
| F5 | Transition heatmaps, mixed | RQ1 | `scripts/a03_transitions.py` | same | `figures/fig_transitions_mixed_image_and_metadata.{png,pdf}` | Confounded pairs; do not interpret as text-only or image-only | appendix caution | no |
| T8 | Point deltas, image-representation pairs | RQ1 | `scripts/a04_effect_sizes.py` | `outputs/master_results.csv` | `tables/tab_deltas_image_representation.tex`; `outputs/condition_deltas.csv` | CORE_PANEL; A2→A4, A2→A5 | yes | with F6 |
| T9 | Point deltas, text-only pairs | RQ1 | `scripts/a04_effect_sizes.py` | same | `tables/tab_deltas_text_only.tex` | CORE_PANEL; A1→A2, A2→A3 | yes | with T6 |
| F6 | Point deltas A2→A4 / A2→A5 | RQ1 | `scripts/a04_effect_sizes.py` | same | `figures/fig_deltas_image_representation.{png,pdf}` | Alignment metrics labeled as Protocol-v2 | yes | yes |
| T10 | Bootstrap CIs on alignment metrics | RQ1 | `scripts/a05_bootstrap.py` | derived table; 2,000 image-cluster replicates, seed 20261002 | `tables/tab_bootstrap_metrics.tex`; `outputs/bootstrap_metric_cis.csv` | CORE_PANEL; 870 images | yes | no (use F7) |
| T11 | Bootstrap CIs, further checkpoints | RQ1 | `scripts/a05_bootstrap.py` | same | `tables/tab_bootstrap_metrics_extended.tex` | Non-CORE usable cells | appendix | no |
| T12 | Bootstrap CIs on paired deltas | RQ1 | `scripts/a05_bootstrap.py` | same | `tables/tab_bootstrap_deltas.tex`; `outputs/bootstrap_delta_cis.csv` | CORE_PANEL; descriptive intervals, no multiplicity correction | yes | with F7 |
| F7 | Forest plot of paired deltas | RQ1 | `scripts/a05_bootstrap.py` | same | `figures/fig_bootstrap_deltas.{png,pdf}` | Spec/Sens/BalAcc labeled alignment | yes | yes |
| T13 | Cross-model agreement by condition | RQ2 | `scripts/a06_agreement.py` | CORE_PANEL decisions, identical IDs | `tables/tab_agreement_by_condition.tex`; `outputs/agreement_*.csv` | 7 models × *N* = 5,747 | yes | optional |
| F8 | Agreement heatmaps (raw / κ) | RQ2 | `scripts/a06_agreement.py` | all models available per condition | `figures/fig_agreement_heatmaps.{png,pdf}` | Upper triangle raw agreement, lower Cohen’s κ | appendix | no (crowded) |
| T14 | Human semantic review counts | RQ4 | `scripts/a07_semantic_review.py` | `outputs/semantic_gt_review/human_review.csv` (SHA verified) | `tables/tab_semantic_review.tex`; `outputs/semantic/semantic_audit.json` | 638 unmatched reviewed; 5,109 matched not reviewed | yes | yes |
| T15 | Geometric reasons for unmatched | RQ4 | `scripts/a07_semantic_review.py` | same + IoU geometry | `tables/tab_unmatched_reasons.tex` | 638 unmatched | yes | yes |
| T16 | VLM verdicts on unmatched by semantic label | RQ4 | `scripts/a07_semantic_review.py` | derived + review | `tables/tab_semantic_by_decision.tex`; `outputs/semantic/verdicts_on_unmatched.csv` | 619 palm / 19 ambiguous / 0 non-palm | yes | optional |
| F9 | Unmatched verdicts, palm vs ambiguous | RQ4 | `scripts/a07_semantic_review.py` | same | `figures/fig_semantic_verdicts.{png,pdf}` | No non-palm panel (count = 0) | yes | yes |
| T17 | Lower-confidence pilot (outside cohort) | RQ4 | `scripts/a07_semantic_review.py` | `outputs/semantic_gt_review/human_confidence_pilot.csv` | `tables/tab_pilot_by_bin.tex` | 400 items, conf 0.10–0.50; no VLM predictions; not prevalence | appendix | no unless discussing limits |
| T18 | IoU-threshold sensitivity | RQ4 | `scripts/a08_iou_sensitivity.py` | stored IoU-sensitivity tables + v2 matcher | `tables/tab_iou_sensitivity.tex`; `outputs/iou_sensitivity/` | Protocol v2 threshold 0.50; semantic labels only for v2-unmatched | yes | optional |
| T19 | Label-usage entropy / majority | RQ3 | `scripts/a10_label_usage_collapse.py` | derived table | `tables/tab_label_usage.tex`; `outputs/label_usage/` | GT-independent; ≥0.99 cuts are descriptive | yes | yes |
| T20 | Per-checkpoint collapse diagnostics | RQ3 | `scripts/a10_label_usage_collapse.py` | same | `tables/tab_label_usage_checkpoint.tex` | Usable conditions per checkpoint | appendix | no |
| F10 | Entropy heatmap | RQ3 | `scripts/a10_label_usage_collapse.py` | same | `figures/fig_label_usage_entropy.{png,pdf}` | GT-independent | yes | yes |
| T21 | A1→A5 change by confidence / area quintile | RQ1 | `scripts/a11_detection_difficulty.py` | derived + index confidence/area | `tables/tab_detection_difficulty.tex`; `outputs/detection_difficulty/` | Paired valid IDs; image-cluster CIs | yes | optional |
| F11 | Changed rate by quintile | RQ1 | `scripts/a11_detection_difficulty.py` | same | `figures/fig_detection_difficulty.{png,pdf}` | Confidence and area are correlated | appendix | no |
| T22 | Within-family size ladder, A1 vs A5 | RQ2 | `scripts/a12_size_ladder.py` | derived + a11 quintiles | `tables/tab_size_ladder.tex`; `outputs/size_ladder/` | Qwen3-VL 2B/4B/8B/32B; InternVL3.5 2B/4B/8B/14B; Qwen2.5-VL 3B/7B (32B not run) | yes | yes |
| C1 | Deterministic qualitative case manifest | RQ1–RQ4 | `scripts/a09_case_selection.py` | CORE_PANEL votes + review labels | `outputs/case_selection/case_manifest.csv` | CORE_PANEL only; ties broken by SHA-256 | yes (qualitative) | use locked IDs only |

## Priority set for the thesis and slides

Use these first. Do not replace them with stronger-looking subsets.

1. Overall decision distributions: T5, F1, F2, T1.
2. A1–A5 comparisons: T6, T8, T9, F3, F4, T12, F7.
3. Paired decision transitions: T6, F3, F4 (not F5 as a main result).
4. Qwen3-VL 2B/4B/8B/32B size ladder: T22. An ordered sequence is not evidence of monotonic scaling.
5. Abstention and decision collapse: T19, F10, T1/T2 (Molmo2-8B A5 is almost all Uncertain).
6. Protocol-v2 alignment with coverage: T3 (Always-Reliable row required).
7. Semantic review: T14, T15, F9.
8. IoU sensitivity: T18.
9. Representative error cases: C1 plus `PRESENTATION_SAMPLES.md`. Do not cherry-pick outside the manifest.

## Do not use as thesis results

- `paper/tables/*.tex` (manuscript stubs; not this rebuild).
- `docs/SEMANTIC_GT_EVALUATION.md` and any semantic-GT “negative” metrics (superseded).
- Alignment specificity presented as semantic specificity.
- Mixed-pair figure F5 interpreted as an image-only or text-only effect.
- Bootstrap intervals as significance tests.
- The 400-item pilot as a cohort prevalence or as VLM performance.
