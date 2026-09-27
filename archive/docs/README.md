# Archived documents

Documents moved here on **2026-09-27** during the Evaluation Protocol v2 migration. They are kept for provenance and are **not** current.

- Any numeric result in these files uses **Evaluation Protocol v1** (case-sensitive `"palm"` ground-truth label; GT+ 4,685 / GT− 1,062; prior 0.815). Full-scale (@5747) numbers are superseded; @1000 and balanced-100 numbers are identical under v2.
- Statuses such as "planned", "running", "next candidate" and "collapsed in every condition" are historical.
- Paths such as `docs/FULL_SCALE_MODEL_COMPARISON.md` inside these files refer to the files' former locations.

**Current sources of truth:** [`docs/EXPERIMENT_RESULTS_CANONICAL.md`](../../docs/EXPERIMENT_RESULTS_CANONICAL.md) (numbers) · [`docs/EXPERIMENT_STATUS_CANONICAL.md`](../../docs/EXPERIMENT_STATUS_CANONICAL.md) (runs) · [`docs/EVALUATION_PROTOCOL.md`](../../docs/EVALUATION_PROTOCOL.md) (protocol).

| File | Former path | What it was | Why archived |
|------|-------------|-------------|--------------|
| [`FULL_SCALE_MODEL_COMPARISON.md`](FULL_SCALE_MODEL_COMPARISON.md) | `docs/` | Advisor-facing A1–A5 @5747 tables and observations | Protocol v1 numbers; content merged into `EXPERIMENT_RESULTS_CANONICAL.md` (v2) |
| [`README_HANDOFF.md`](README_HANDOFF.md) | `docs/` | README rewrite hand-off brief | Protocol v1 numbers; README rewritten for v2 |
| [`ADVISOR_RESEARCH_STATUS_20260913.md`](ADVISOR_RESEARCH_STATUS_20260913.md) | `docs/` | Dated advisor status snapshot (2026-09-13) | Superseded snapshot |
| [`GEMMA_IMPLEMENTATION_REPORT.md`](GEMMA_IMPLEMENTATION_REPORT.md) | `docs/` | Gemma 3 integration report (pre-smoke) | Integration complete; Gemma 3 stopped after A1@1000. Its links to the deleted `GEMMA_INTEGRATION_PLAN.md` are dead by design |
| [`MULTI_MODEL_INTEGRATION_PLAN.md`](MULTI_MODEL_INTEGRATION_PLAN.md) | `docs/` | July 2026 multi-model design plan | Plan executed; statuses superseded |
| [`VLM_MODEL_SELECTION_SURVEY.md`](VLM_MODEL_SELECTION_SURVEY.md) | `docs/` | Early VLM candidate survey | Superseded by `MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md` |
| [`REPOSITORY_CLEANUP_REPORT.md`](REPOSITORY_CLEANUP_REPORT.md) | `docs/` | June/July 2026 cleanup record | Historical record |
| [`phase2_plan.md`](phase2_plan.md) | `docs/` | Early phase-2 project plan | Superseded plan |

Deleted in the same migration (verified to contain no results or required instructions): `docs/GEMMA_INTEGRATION_PLAN.md` (Gemma 3 pre-implementation plan; its design decisions are recorded in `GEMMA_IMPLEMENTATION_REPORT.md` and the code). Recoverable from git history.
