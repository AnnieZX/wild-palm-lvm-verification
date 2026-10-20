# Presentation sample IDs (deterministic)

Selected from `outputs/case_selection/case_manifest.csv` (a09; CORE_PANEL votes;
ties broken by `sha256("20261002:<category>:<sample_id>")`). No ad-hoc cherry-picking.
All five IDs are in the case manifest, so the static demo exporter includes them.

Demo deep-link: `/?d=sample_XXXXXX` (static or live). Images are the stored A1–A5
condition files plus the raw patch. Predictions are the saved inference JSON, not
recomputed.

| Role | sample_id | image_id | LabelMe | Semantic | Why this ID |
|---|---|---|---|---|---|
| A1–A5 decision change | `sample_003365` | `100_0003_0074_2` | unmatched (max IoU 0.0305, partial_overlap) | palm | `strong_A2_to_A5_change` rank 1: 7/7 CORE models change A2→A5 |
| Unmatched detection reviewed as palm | `sample_004284` | `100_0003_0194_6` | unmatched (IoU 0.0, no_overlap) | palm | `consensus_unreliable` rank 4; LabelMe-unmatched, human palm |
| Abstention-heavy | `sample_000411` | `100_0003_0012_1` | matched (IoU 0.9117) | not reviewed | `abstention_heavy` rank 1 (23 Uncertain votes / 35 CORE cells) |
| Qwen3-VL 2B/4B/8B/32B comparison | `sample_003020` | `100_0003_0068_5` | matched (IoU 0.69) | not reviewed | `high_disagreement` rank 1; A5 labels differ across the four Qwen3-VL checkpoints |
| Unmatched palm + Qwen3-VL split | `sample_001333` | `100_0003_0027_7` | unmatched (IoU 0.0, no_overlap) | palm | `abstention_heavy` rank 2; three distinct A5 labels on the Qwen3-VL ladder |

## Stored Qwen3-VL decisions (from `derived/sample_decisions.csv.gz`)

### sample_003365

| Checkpoint | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Qwen3-VL-2B | Reliable | Reliable | Reliable | Reliable | Reliable |
| Qwen3-VL-4B | Unreliable | Unreliable | Unreliable | Unreliable | Uncertain |
| Qwen3-VL-8B | Reliable | Reliable | Reliable | Reliable | Uncertain |
| Qwen3-VL-32B | Reliable | Reliable | Reliable | Unreliable | Uncertain |

CORE-panel vote strings A1–A5: `URXRRRR` / `URXRRRR` / `URXUXRR` / `XRXURXR` / `XUUUXXX`.

### sample_004284

| Checkpoint | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Qwen3-VL-2B | Reliable | Reliable | Reliable | Uncertain | Uncertain |
| Qwen3-VL-4B | Unreliable | Unreliable | Unreliable | Unreliable | Unreliable |
| Qwen3-VL-8B | Unreliable | Unreliable | Unreliable | Unreliable | Unreliable |
| Qwen3-VL-32B | Unreliable | Unreliable | Unreliable | Unreliable | Uncertain |

CORE-panel vote strings A1–A5: `UXXXXXX` / `UXXXXXX` / `UXXXXXX` / `XXXUXXX` / `XXXUXXX`.

### sample_000411

| Checkpoint | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Qwen3-VL-2B | Reliable | Reliable | Reliable | Uncertain | Reliable |
| Qwen3-VL-4B | Uncertain | Uncertain | Uncertain | Uncertain | Uncertain |
| Qwen3-VL-8B | Reliable | Reliable | Uncertain | Uncertain | Uncertain |
| Qwen3-VL-32B | Unreliable | Unreliable | Uncertain | Uncertain | Uncertain |

CORE-panel vote strings A1–A5: `URUUUXR` / `URUUUXR` / `UUUUUXR` / `UUUUUXR` / `UUUUUXR`.

### sample_003020

| Checkpoint | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Qwen3-VL-2B | Reliable | Reliable | Reliable | Reliable | Reliable |
| Qwen3-VL-4B | Unreliable | Unreliable | Unreliable | Unreliable | Uncertain |
| Qwen3-VL-8B | Reliable | Reliable | Reliable | Reliable | Uncertain |
| Qwen3-VL-32B | Unreliable | Unreliable | Unreliable | Reliable | Unreliable |

CORE-panel vote strings A1–A5: `URXURXR` / `URXURXR` / `URXUXXR` / `URXUXXR` / `RUUUXXR`.

### sample_001333

| Checkpoint | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Qwen3-VL-2B | Reliable | Reliable | Reliable | Reliable | Reliable |
| Qwen3-VL-4B | Uncertain | Uncertain | Uncertain | Uncertain | Unreliable |
| Qwen3-VL-8B | Uncertain | Uncertain | Uncertain | Reliable | Uncertain |
| Qwen3-VL-32B | Uncertain | Uncertain | Uncertain | Uncertain | Uncertain |

CORE-panel vote strings A1–A5: `UUUUXRR` / `UUUUURR` / `UUUUUXX` / `URUUXXR` / `RUXUUXR`.

## How to present

- Say what changed: text-only (A1–A3, same image) versus image-representation
  (A2/A4/A5, image and image-description text). Do not call A2→A5 image-only.
- For `sample_004284` and `sample_001333`: unmatched ≠ non-palm. The human label is palm.
- For `sample_000411`: matched detections were never semantically reviewed.
- For the size ladder: one item is an illustration, not a scaling proof.
