# Thesis analysis layer

Descriptive analysis of the frozen full-cohort (N = 5,747) VLM verification runs. Every
script **reads** `outputs/` and writes only under `paper/analysis/`. No inference is rerun
and no raw prediction, evaluation, or review file is modified.

```bash
bash paper/analysis/run_all.sh                 # full rebuild (~1 min)
bash paper/analysis/run_all.sh --no-recompute  # skip the in-memory IoU re-matching in a08
```

Environment used: Python 3.9, pandas 2.3.3, numpy 2.0.2, matplotlib 3.9.4.

## Run order and outputs

| Script | Purpose | Main outputs |
|---|---|---|
| `a00_audit.py` | Integrity checks; builds the sample-level table | `outputs/audit/*`, `derived/sample_decisions.csv.gz` |
| `a01_master_table.py` | One row per model × condition | `outputs/master_results.csv`, `tables/tab_master_{behavior,alignment}.tex` |
| `a02_decision_distributions.py` | R/U/Ur rates | `outputs/decision_distribution_*.csv`, `figures/fig_decision_*` |
| `a03_transitions.py` | 3×3 sample-level transition matrices | `outputs/transition_*.csv`, `figures/fig_transitions_*` |
| `a04_effect_sizes.py` | Point deltas between conditions | `outputs/condition_deltas.csv`, `tables/tab_deltas_*.tex` |
| `a05_bootstrap.py` | Image-level cluster bootstrap CIs | `outputs/bootstrap_*.csv`, `figures/fig_bootstrap_deltas` |
| `a06_agreement.py` | Cross-model agreement per condition | `outputs/agreement_*.csv`, `figures/fig_agreement_heatmaps` |
| `a07_semantic_review.py` | Human semantic review (RQ4) and pilot | `outputs/semantic/*`, `tables/tab_semantic_*.tex`, `tab_unmatched_reasons.tex`, `tab_pilot_by_bin.tex` |
| `a08_iou_sensitivity.py` | Provenance audit and use of IoU-threshold tables | `outputs/iou_sensitivity/*`, `tables/tab_iou_sensitivity.tex` |
| `a09_case_selection.py` | Deterministic qualitative case manifest | `outputs/case_selection/*` |
| `a10_label_usage_collapse.py` | GT-independent label usage, entropy, majority share, collapse diagnostics | `outputs/label_usage/*`, `tables/tab_label_usage*.tex`, `figures/fig_label_usage_entropy` |
| `a11_detection_difficulty.py` | A1→A5 / A2→A5 change by confidence and box-area quintile; Q1−Q5 bootstrap contrasts; data-derived verdicts | `outputs/detection_difficulty/*`, `tables/tab_detection_difficulty.tex`, `figures/fig_detection_difficulty` |
| `a12_size_ladder.py` | Within-family size ladders (Qwen3-VL, InternVL3.5, Qwen2.5-VL), A1 vs A5 | `outputs/size_ladder/*`, `tables/tab_size_ladder.tex` |

`common.py` holds the run registry (`RUNS`), the seven-model panel (`CORE_PANEL`), the
size ladders (`LADDERS`), the condition-status policy (`CONDITION_POLICY`), paths, metric
definitions, seeds and the shared bootstrap weight generator.

## Condition status and which analyses use what

`a00` assigns every model × condition one status:

- **full_clean**: 5,747 files and every integrity check passes (`analysed = True`).
- **partial**: 5,747 files; the only defects are parse errors, recorded consistently
  (empty decision, `results_index` = `parse_error` on exactly those IDs, empty Protocol-v2
  label, stored metrics = recomputation over the valid rows). Valid rows only; nothing is
  repaired, re-parsed or imputed; the protocol is unchanged.
  - Qwen3-VL-2B A3 and A5: one parse error each (512-token repetition loop), N = 5,746.
  - InternVL3.5-2B A1–A5: 4 / 211 / 157 / 22 / 93 parse errors; every parseable decision
    is Uncertain (degenerate checkpoint; see a10). Not re-run.
- **failed / incomplete / not_run**: excluded.

The derived sample table holds valid rows of full_clean and partial cells.

| Analysis | Cells used | N handling |
|---|---|---|
| a01, a02, a10 | full_clean + partial | per-cell valid N reported; partial marked † |
| a03, a05 transitions, a11, a12 | full_clean + partial | pairs use detections valid in both conditions; N per pair reported |
| a04, a05 metric deltas | full_clean + partial | marginal metrics; `n_from`/`n_to`/`same_n` columns |
| a06, a07, a08, a09 | full_clean only | identical N = 5,747 |

Thesis tables built on the original seven-model panel (`tab_master_*`, `tab_transitions_summary`,
`tab_deltas_*`, `tab_bootstrap_*`, `tab_agreement_by_condition`, case manifest) stay on
`CORE_PANEL`, so their values are unchanged; further checkpoints are in `*_extended.tex` and in
the CSVs. Collapse cut-offs in a10 (majority ≥ 0.99 / ≥ 0.95) are descriptive diagnostics, not
scientific thresholds.

## Constructs (kept separate everywhere)

1. **Behaviour**: reference-free R/U/Ur rates, coverage = 1 − U rate.
2. **Protocol v2 IoU alignment**: greedy one-to-one per image, IoU ≥ 0.5 with LabelMe
   `palm` boxes; 5,109 *matched* / 638 *unmatched*. Columns are named `align_R_matched`
   (legacy TP), `align_R_unmatched` (legacy FP), `align_Ur_matched` (legacy FN),
   `align_Ur_unmatched` (legacy TN). Uncertain is excluded from the binary metrics.
   "Unmatched" is an IoU outcome, not a semantic class.
3. **Human semantic review**: the 638 unmatched detections only (619 palm / 19 ambiguous /
   0 non_palm). Matched detections were never reviewed. The 400-item lower-confidence pilot
   lies outside the cohort, has no VLM predictions, and is analysed separately.

## Condition pairs

From the executed prompts (`outputs/verification_ablation_5747/*/prompts`), not the README:

- `text_only` (A1→A2, A2→A3, A1→A3): identical image; the metadata block, instruction
  lines and condition label change.
- `image_representation` (A2→A4, A2→A5, A4→A5): identical confidence text; the image,
  the "Input image" description paragraph and the condition label change (A2→A4/A5 also
  change one instruction line). **Not image-only.**
- `mixed_image_and_metadata`: A1→A4, A1→A5, A3→A4, A3→A5.

## Bootstrap procedure (a05)

- Unit: the source image (870 images; 1–20 detections per image, median 6).
- 2,000 replicates; `numpy.random.Generator(PCG64(20261002))`. Each replicate draws 870
  images with replacement; image multiplicities become weights on per-image cell counts.
- The **same** weight vector is applied to every model and condition in a replicate, so
  condition and model contrasts are paired.
- Intervals are 2.5/97.5 percentile CIs. They describe resampling variability only and
  are not used for significance claims.
- a11 and a12 use the same generator (`common.image_weights`), seed and image order, so
  replicate *r* draws the same images in a05, a11 and a12.

## Agreement (a06)

Pairwise raw agreement, chance agreement and Cohen's κ (3-class) per condition. The
pairwise CSV also gives **κ_max** (the highest κ the two marginals allow), because
Phi-4-multimodal and Gemma-4-12B never answer Uncertain and Reliable dominates several
models; high raw agreement with κ ≈ 0.1 is the expected prevalence effect. Fleiss' κ assumes
interchangeable raters; it is reported only as a summary next to Light's κ (mean pairwise
Cohen). The 7-model summary keeps the model set constant across A1–A5;
`agreement_by_condition_all_clean_complete.csv` repeats it for every model with five full_clean
conditions. `agreement_by_condition_all_models.csv` uses every model available per condition.

## Case selection (a09)

Fixed score per category, descending, ties broken by
`sha256("20261002:<category>:<sample_id>")`. `n_at_max_score` shows when a category is a
plateau, i.e. a deterministic draw rather than a list of the most extreme cases. See the
script docstring for all rules.

## Excluded or pending runs

- `qwen2_5_vl/20260708_0020/A5`: cancelled wrong-path job, excluded.
- MiniCPM-V-4.5: A1–A3 full_clean; A4/A5 queued.
- Molmo2-8B: A1 full_clean; A2–A5 queued.
- Qwen3-VL-32B: A1–A5 submitted 2026-10-04; audited as not_run until complete.
- Qwen2.5-VL-32B: qualification probe only; listed as "not run" on the size ladder.
