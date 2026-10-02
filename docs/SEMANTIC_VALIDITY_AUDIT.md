# Semantic Validity Audit — Canonical Record

**Status:** official · **Audit date:** 2026-10-01 · **Reviewer:** one human reviewer (`zixiao`)

This is the canonical record of the human semantic-validity audit. It is the only source for
semantic (palm / non-palm) statements in this project. It does not change
[Evaluation Protocol v2](EVALUATION_PROTOCOL.md), any stored prediction, or any Protocol-v2 metric.

The project distinguishes three layers:

| Layer | Question | Source |
|---|---|---|
| **Primary** | How do A1–A5 input conditions change VLM verification behavior (Reliable / Uncertain / Unreliable, coverage, consistency)? | Stored VLM predictions only |
| **Secondary** | Do VLM decisions agree with LabelMe annotation alignment? | Protocol v2 (greedy one-to-one, IoU ≥ 0.5) |
| **Audit** (this document) | Are the detections semantically palms? | Human visual review |

---

## A. Why the audit was necessary

Protocol v2 labels a YOLO detection GT+ if it is matched to a LabelMe `palm` box at IoU ≥ 0.5
(greedy one-to-one) and GT− otherwise. Historical documents treated GT− as "detector false
positive" and read specificity (Unreliable on GT−) as the ability to reject non-palms.

That reading assumes LabelMe annotations are exhaustive and well-aligned. They are not documented
as such, and a detection can be GT− because the palm has no LabelMe box, because its LabelMe box
overlaps the YOLO box at IoU < 0.5, or because a better-overlapping duplicate detection took the
LabelMe box. The audit tests the assumption directly by looking at the detections.

## B. Protocol-v2 LabelMe-unmatched audit (official)

Population: **all 638** LabelMe-unmatched (GT−) detections of the canonical 5,747-detection cohort
(YOLO confidence ≥ 0.5). The row set equals the GT− set of every full-cohort Protocol-v2 evaluation
file (60 files checked); no duplicates; no missing labels.

| Semantic label | Count | Share of 638 |
|---|--:|--:|
| palm | **619** | 97.0% |
| ambiguous | **19** | 3.0% |
| non_palm | **0** | 0.0% |

Subsets used in historical analyses:

| Subset of the 638 | n | palm | ambiguous | non_palm |
|---|--:|--:|--:|--:|
| Stage-1 balanced-100 qualification "negatives" | 50 | 45 | 5 | 0 |
| A1@1000 slice (`sample_000001`–`sample_001000`) GT− | 72 | 65 | 7 | 0 |

Review conditions: the reviewer saw only the raw patch and the target YOLO box; no LabelMe
annotation, YOLO confidence, IoU, other detection or VLM output was shown. Labels:
`palm`, `non_palm`, `ambiguous`. Labelling window 2026-10-01 15:28–15:41 UTC; 642 logged events,
4 explicit label changes.

## C. Lower-confidence stratified pilot

Purpose: check whether semantic non-palms exist **below** the ≥ 0.5 verification cohort, i.e.
whether the population contains a non-palm class at all. This pilot is **not** part of the
5,747-detection cohort and has no VLM predictions.

Design: blind, confidence-stratified restricted random sample of YOLO detections, 100 per bin
(patch / parent-frame caps; selection seed 20261001, review-order seed 20261002). The reviewer saw
only the patch and the box, in a shuffled order carrying no confidence information.

| YOLO confidence bin | n | palm | non_palm | ambiguous |
|---|--:|--:|--:|--:|
| [0.10, 0.20) | 100 | 76 | 4 | 20 |
| [0.20, 0.30) | 100 | 68 | 3 | 29 |
| [0.30, 0.40) | 100 | 85 | 1 | 14 |
| [0.40, 0.50) | 100 | 89 | 1 | 10 |
| **Overall** | **400** | **318** | **9** | **73** |

## D. Warning: the pilot's overall percentages are not prevalence

The pilot sampled **equal 100-per-bin** strata, while the bins contain very different numbers of
detections (1,891 / 892 / 588 / 527 from lowest to highest bin). The raw overall shares
(318 / 9 / 73 of 400) are therefore **not** estimates of the natural prevalence of palm, non-palm or
ambiguous among lower-confidence candidates. Any population estimate must re-weight by stratum
(and account for the patch / frame caps). Per-bin counts are the reportable quantity.

## E. Interpretation

- **LabelMe annotation alignment and human semantic palm validity are different constructs.**
  Among LabelMe-unmatched detections at confidence ≥ 0.5, the reviewer found no non-palm.
- Protocol-v2 GT− should be read as **LabelMe-unmatched**, not as "detector false positive".
  Protocol-v2 "FP" and "TN" are alignment quantities.
- Protocol-v2 specificity is **alignment specificity** (share of decided LabelMe-unmatched
  detections classified Unreliable). High alignment specificity on this cohort mostly means the
  model answers Unreliable on detections that a human judged to be palms; it is not evidence of
  non-palm rejection.
- Historical qualification gates (balanced-100 Spec / BalAcc) were **alignment-based**; their 50
  "negatives" were 45 palm / 5 ambiguous / 0 non-palm.
- Non-palm detections do occur at lower YOLO confidence (9 confirmed in the pilot, concentrated in
  the lowest bins), so the absence of non-palms is a property of the ≥ 0.5 cohort, not of the
  detector at all thresholds.
- The primary analysis of this project is therefore VLM **behavior** across A1–A5, which needs no
  ground truth. Protocol-v2 alignment results are secondary.

## F. Limitations

- **Single reviewer.** No inter-rater agreement was measured.
- **Ambiguous retained.** Ambiguous cases were kept as their own category rather than forced into
  palm / non-palm; they are excluded from any binary statement.
- **Visual semantic review only.** The question was "is this a palm?"; box localization quality
  (tightness, partial coverage, duplicates) was not assessed, so `palm` does not mean the box is
  well-localized. The VLM's Unreliable verdict also covers "the detection is clearly incorrect",
  which this audit does not score.
- **No semantic specificity for the ≥ 0.5 cohort.** With 0 confirmed non-palms, semantic
  non-palm specificity cannot be estimated for the 5,747-detection cohort, and none is claimed.
- **Small non-palm sample.** The pilot contains only **9** confirmed non-palms.
- **Population-aware 638 review.** All 638 items were LabelMe-unmatched; no LabelMe-matched
  controls were interleaved. A 100-row matched-positive blind QC sample exists but has not been
  labelled. The review pass was fast (638 items in about 13 minutes).
- **LabelMe-matched detections were not reviewed.** The 5,109 GT+ detections are not semantically
  audited here.

## G. Provenance

Official label files (active copies; must not be edited):

| File | Rows | SHA-256 |
|---|--:|---|
| `outputs/semantic_gt_review/human_review.csv` | 638 | `b2a0219df9b3310661752e2dcbcd5d9b5bb9501dcdc1d07cfeb5f7284f478485` |
| `outputs/semantic_gt_review/human_review.log.jsonl` | 642 events | `a1a4d62fe8e1ccfc20edb785eae7efd33e58604913bde0c398675866ad7a0691` |
| `outputs/semantic_gt_review/human_confidence_pilot.csv` | 400 | `9a25cd38a1923e135b9ecef83b16397e8bf6254b1a9ba78948b77d01fdb97705` |
| `outputs/semantic_gt_review/human_confidence_pilot.log.jsonl` | — | `43fe8dceb9a48aaac435cd0c6f0825678075b9f61a9854f1fac687b4677bad6c` |
| `outputs/semantic_candidate_audit/semantic_pilot_manifest.csv` (blind) | 400 | `d89953d6812ccc9efa80ee7a3ea94e718c9cde2e450ce6082fb8d823fec42a39` |
| `outputs/semantic_candidate_audit/semantic_pilot_reference.csv` (sealed during review) | 400 | `4e02f77ab4ea44b026267833314cba688cea7e204da3eaa0623d2e26321ca062` |

Read-only archive snapshots under `outputs/semantic_gt_review/archive/` (each has `README.md` and
`SHA256SUMS`; verify with `sha256sum -c SHA256SUMS` inside the directory):

| Snapshot | Content | SHA-256 of its `SHA256SUMS` |
|---|---|---|
| `official_638_unmatched_review_completed_20261001T215529Z/` | Official 638-row review, change log, 25 autosave backups | `9f6662b0c3501175cdc48bcd731ef8972710493c9a7b36042fc6a0c70b0120df` |
| `confidence_pilot_official_completed_20261001T172808Z/` | Official 400-row pilot, change log, autosave backups | `6d39e6bd156e5fa43d45526fa30ab5525f21752e31ba4fff53b58237db4f4608` |
| `confidence_pilot_familiarization_20261001T165738Z/` | 148 practice labels entered before criteria were standardized — **must not be analysed** | `ea00024d7cb2d372552a0c065e72d9e059fc78ee2ee8a42a43c498f36d65718a` |

Tools: blind review app `scripts/semantic_review_app.py` (`src/semantic_review/`); pilot sampling
and analysis `outputs/semantic_candidate_audit/` (`build_semantic_candidate_audit.py`,
`analyze_confidence_pilot.py`).

The earlier semantic-GT pipeline (`src/evaluation/semantic_gt.py`, `scripts/evaluate_semantic_gt.py`,
`outputs/semantic_gt_evaluation/`, [SEMANTIC_GT_EVALUATION.md](SEMANTIC_GT_EVALUATION.md)) is
**SUPERSEDED / NOT FOR CURRENT RESULTS**: it never read these label files.
