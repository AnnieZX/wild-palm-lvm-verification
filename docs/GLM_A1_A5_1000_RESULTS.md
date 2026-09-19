# GLM-4.6V-Flash results

**A1–A5 @1000**

Model: **GLM-4.6V-Flash**

Qualification/comparison subset; metrics use the same verifier evaluation protocol as Qwen.

Presentation page (white background, matching Qwen table layout):
`docs/figures/glm_a1_a5_1000_results.html`

All values below were recomputed from evaluation CSVs and cross-checked against stored `*_metrics.json` (Spec / BalAcc derived from TP/FP/TN/FN).

## Metrics table

| Abl. | N | R / U / Ur | TP | FP | TN | FN | Prec | Rec | F1 | Acc | Spec | BalAcc |
|------|--:|------------|---:|---:|---:|---:|-----:|----:|---:|----:|-----:|-------:|
| A1 | 1000 | 641 / 8 / 351 | 611 | 30 | 42 | 309 | 0.9532 | 0.6641 | 0.7828 | 0.6583 | 0.5833 | 0.6237 |
| A2 | 1000 | 650 / 7 / 343 | 620 | 30 | 42 | 301 | 0.9538 | 0.6732 | 0.7893 | 0.6667 | 0.5833 | 0.6283 |
| A3 | 1000 | 638 / 11 / 351 | 618 | 20 | 51 | 300 | 0.9687 | 0.6732 | 0.7943 | 0.6764 | 0.7183 | 0.6958 |
| A4 | 1000 | 676 / 7 / 317 | 653 | 23 | 49 | 268 | 0.9660 | 0.7090 | 0.8178 | 0.7069 | 0.6806 | 0.6948 |
| A5 | 1000 | 400 / 179 / 421 | 392 | 8 | 36 | 385 | 0.9800 | 0.5045 | 0.6661 | 0.5213 | 0.8182 | 0.6613 |

## Key finding

GLM remains non-collapsed across A1–A5 and reproduces the Qwen A5 tradeoff: higher precision but substantially lower recall and F1 (best A4 F1 = 0.8178; A4→A5: F1 0.8178→0.6661, Prec 0.9660→0.9800, Rec 0.7090→0.5045).

## Source files (evaluation CSVs)

| Abl. | Source |
|------|--------|
| A1 | `outputs/evaluation/glm_4_6v_flash/20260913_glm46v_flash_A1_1000/A1/A1_evaluation.csv` |
| A2 | `outputs/evaluation/glm_4_6v_flash/20260914_glm46v_flash_A2A5_1000/A2/A2_evaluation.csv` |
| A3 | `outputs/evaluation/glm_4_6v_flash/20260914_glm46v_flash_A2A5_1000/A3/A3_evaluation.csv` |
| A4 | `outputs/evaluation/glm_4_6v_flash/20260914_glm46v_flash_A2A5_1000/A4/A4_evaluation.csv` |
| A5 | `outputs/evaluation/glm_4_6v_flash/20260914_glm46v_flash_A2A5_1000/A5/A5_evaluation.csv` |

Protocol: Reliable = predicted positive; Unreliable = predicted negative; Uncertain excluded from Prec/Rec/F1/Acc/Spec/BalAcc.
