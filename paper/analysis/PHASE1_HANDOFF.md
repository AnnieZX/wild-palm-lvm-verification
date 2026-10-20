# Phase 1 handoff — for Phase 2 manuscript writing

Do not start from `paper/tables/*.tex` or the `paper/sections/*.tex` TODO stubs unless
you are writing new prose. Canonical numbers and generated tables/figures are below.

**Do not** rerun inference, submit Slurm, edit raw `outputs/` predictions, edit human
review files, or use the superseded semantic-GT evaluation pipeline.

## Canonical sources

| What | Path |
|---|---|
| Run registry / CORE_PANEL / ladders | `paper/analysis/scripts/common.py` |
| Integrity audit | `paper/analysis/outputs/audit/AUDIT_REPORT.md`, `model_inventory.csv` |
| Sample-level decisions | `paper/analysis/derived/sample_decisions.csv.gz` (430,536 valid rows) |
| Master metrics | `paper/analysis/outputs/master_results.csv` |
| Figure/table inventory | `paper/analysis/FIGURE_TABLE_INVENTORY.md` |
| Presentation sample IDs | `paper/analysis/PRESENTATION_SAMPLES.md` |
| RQ notes (status updated 2026-10-19; CORE_PANEL bullets still 2026-10-02) | `paper/analysis/RQ_EVIDENCE_MATRIX.md` |
| Analysis README | `paper/analysis/README.md` |
| Evaluation protocol | `docs/EVALUATION_PROTOCOL.md` |
| Semantic validity | `docs/SEMANTIC_VALIDITY_AUDIT.md` |

## Constructs (keep separate in the manuscript)

1. Ground-truth-free R/U/Ur behaviour. Coverage = (R+Ur)/N.
2. Protocol-v2 LabelMe alignment (greedy 1-1, IoU ≥ 0.5): 5,109 matched / 638 unmatched.
   Acc/F1 exclude Uncertain; always report coverage and the Always-Reliable baseline
   (Acc 0.889, F1 0.941). Alignment specificity ≠ semantic specificity.
3. Human semantic review of unmatched detections only: 619 palm / 19 ambiguous / 0 non-palm.
   Semantic non-palm specificity is not estimable. Matched detections were never reviewed.

A1–A3 are text-only (identical image). A2/A4/A5 are image-representation contrasts
(image *and* image-description text). They are not image-only.

Primary thesis tables stay on `CORE_PANEL` (7 models). Qwen3-VL-32B, MiniCPM-V-4.5,
Molmo2-8B and the other later checkpoints are in `*_extended` tables and `tab_size_ladder.tex`.

## Qwen3-VL-32B (full_clean A1–A5)

Experiment: `outputs/verification/qwen3_vl_32b/20261004_qwen3_vl_32b_A1A5_5747`.
5,747 unique IDs per condition; 0 parse errors; 0 inference errors.

| Cond | R | U | Ur |
|---|--:|--:|--:|
| A1 | 3,926 | 189 | 1,632 |
| A2 | 4,006 | 227 | 1,514 |
| A3 | 4,087 | 395 | 1,265 |
| A4 | 3,638 | 339 | 1,770 |
| A5 | 467 | 3,060 | 2,220 |

## Generated thesis tables

Use `paper/analysis/tables/` (auto-generated). Priority:

- `tab_master_behavior.tex`, `tab_decision_distribution.tex`
- `tab_transitions_summary.tex`, `tab_deltas_text_only.tex`, `tab_deltas_image_representation.tex`
- `tab_bootstrap_deltas.tex`, `tab_bootstrap_metrics.tex`
- `tab_size_ladder.tex`
- `tab_master_alignment.tex` (Always-Reliable row required)
- `tab_semantic_review.tex`, `tab_unmatched_reasons.tex`, `tab_semantic_by_decision.tex`
- `tab_iou_sensitivity.tex`
- `tab_label_usage.tex`

Extended / appendix: `*_extended.tex`, `tab_pilot_by_bin.tex`, `tab_detection_difficulty.tex`.

## Generated thesis figures

Use `paper/analysis/figures/` (`png` + `pdf`):

- `fig_decision_distribution_stacked`, `fig_decision_rates_by_condition`
- `fig_transitions_text_only`, `fig_transitions_image_representation`
- `fig_deltas_image_representation`, `fig_bootstrap_deltas`
- `fig_label_usage_entropy`, `fig_semantic_verdicts`
- `fig_transitions_mixed_image_and_metadata` — appendix only; do not interpret as image-only

## Qualitative cases

Deterministic manifest: `paper/analysis/outputs/case_selection/case_manifest.csv`.
Locked presentation IDs: `sample_003365`, `sample_004284`, `sample_000411`,
`sample_003020`, `sample_001333`. Demo deep-link: `/?d=sample_XXXXXX`.

## Demo (already rebuilt; do not redeploy)

Real demo: `demo/web/` + `demo/backend/`. Mock `demo/frontend/` was not used.
Static snapshot: `demo/web/public/snapshot/` (gitignored; 300 of 5,747 explorer
detections, generated 2026-10-20). Production build: `demo/web/dist/` (gitignored).
Preview locally with `cd demo/web && VITE_STATIC=1 npx vite preview`.
No GPU / no live inference in static mode.

## What Phase 2 should write

Honors-thesis prose in `paper/sections/` using the generated tables/figures above.
Do not overwrite generated `paper/analysis/tables/` by hand. Do not invent a
stronger interpretation than `RQ_EVIDENCE_MATRIX.md` and the table captions allow.
