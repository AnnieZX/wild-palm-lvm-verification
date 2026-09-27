# Wild Palm LVM Verification

**CS Honors Thesis** · Annie Luo · Mentor: Fan Yang · Wake Forest University · 2026

Second-stage **Vision-Language Model (VLM) verification** of fixed YOLO wild-palm detections in UAV orthomosaic imagery. This repository does **not** run detection; LabelMe ground truth is used only for **evaluation**.

---

## Research Question

Can modern VLMs verify YOLO palm detections by accepting true positives and rejecting false positives, and how does available visual context (ablations **A1–A5**) change verification reliability?

Decisions are closed-set: **Reliable** (accept) · **Uncertain** (abstain) · **Unreliable** (reject). All compared models share a frozen dataset, prompts, parser, and evaluator.

**Binary metrics** (Acc / Sens / Spec / F1) use only Reliable vs Unreliable; Uncertain is excluded. Always interpret them **alongside** the three-way R / U / Ur counts. Two post-hoc descriptive rates (not part of the frozen evaluator) clarify coverage:

| Rate | Definition |
|------|------------|
| **Decision Coverage** | (Reliable + Unreliable) / N |
| **Abstention Rate** | Uncertain / N |

---

## Experimental Protocol

| Item | Definition |
|------|------------|
| Input | Fixed YOLO detections (N = **5,747**) |
| GT match | LabelMe `palm` boxes; greedy 1–1; **IoU ≥ 0.5** |
| GT prior | GT+ = 4,685 · GT− = 1,062 · always-Reliable Acc ≈ **0.815** |
| Ablations | **A1** overlay · **A2** +conf · **A3** +geometry · **A4** dual panel · **A5** crop only |
| Fairness | Frozen parser + evaluator; model-native vision/chat only |

Details: [`docs/ABLATION_STUDY.md`](docs/ABLATION_STUDY.md) · [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md)

---

## Models Evaluated

*Run status* (what was executed) and *scientific outcome* (what it shows) are reported separately.

| Model | Scale | Run status | Scientific outcome | Notes |
|-------|-------|-----------|--------------------|-------|
| **Qwen2.5-VL-7B** | A1–A5 @5747 | **Complete** | Useful verifier | PRIMARY baseline; three-way labels |
| **Qwen3-VL-8B** | A1–A5 @5747 | **Complete** | Useful verifier | Within-family support; A5 Uncertain-heavy |
| **GLM-4.6V-Flash** | A1–A5 @5747 | **Complete** | Useful verifier | PRIMARY; strong Unreliable / Spec |
| **Phi-4 Multimodal** | A1–A5 @5747 | **Complete** | Useful verifier | PRIMARY; **0 Uncertain** all conditions |
| **InternVL3.5-8B-HF** | A1–A5 @5747 | **Complete** | Abstention-heavy collapse | U 39–71%; Ur ≤ 2.7%; Spec 0.016–0.144 |
| **Gemma 4 12B IT** | A1–A5 @5747 | **Complete** | Reliable-heavy collapse | R 91.5–97.4%; Spec 0.068–0.236; skipped normal qualification gates |
| InternVL3-8B | balanced-100 | Qualification failed / stopped | Technical failure | 13/100 parse failures; Spec 0 |
| MiniCPM-V-4.5 | A1 @5747 | A1 only | Reliable-heavy collapse | R/U/Ur 5557/0/190; Spec 0.078 |
| Molmo2-8B | A1 @5747 | A1 only | Abstention-heavy collapse | R/U/Ur 1209/4537/1; Spec 0 |
| LLaVA-OneVision | A1 @1000 | A1 only | Reliable-heavy collapse (single-class) | 100% Reliable |
| Gemma 3 12B | A1 @1000 | A1 only | Reliable-heavy collapse (single-class) | 100% Reliable |

Next candidate / not yet evaluated: Llama-3.2-11B-Vision-Instruct.

Authoritative inventory: [`docs/EXPERIMENT_STATUS_CANONICAL.md`](docs/EXPERIMENT_STATUS_CANONICAL.md)

---

## Key Results

- **Six models completed** full A1–A5 @5747. Four are useful verifiers (Qwen2.5-VL, Qwen3-VL, GLM-4.6V-Flash, Phi-4); two collapsed at full scale (InternVL3.5 abstention-heavy, Gemma 4 Reliable-heavy).
- **Context matters:** e.g. Qwen2.5 A3 Spec ≈ 0.02 vs A5 Spec ≈ 0.72; Qwen3 A5 becomes highly abstaining (U=3948, Spec ≈ 0.81).
- **Collapse cases:** LLaVA and Gemma predict Reliable on all 1000 A1 samples (Acc = 0.928 = class prior; Spec = 0).
- **Partial collapse is two-sided:** MiniCPM is near–always-Reliable; Molmo2 abstains on **4537/5747** (Abstention ≈ 0.79). Molmo2 binary F1=0.962 is **misleading** — Uncertain is excluded, so F1 is computed on only ~1210 scored predictions.
- **InternVL3 → InternVL3.5:** same balanced-100 gate; parse 87%→100%, Spec 0→0.20, BalAcc 0.50→0.60; Uncertain-heavy. Version and HF API both changed — do not over-attribute.
- **InternVL3.5 full-scale A1–A5** completed but is abstention-heavy: Uncertain 39–71%, Spec ≤ 0.144 on every condition. Its Acc ≈ 0.85–0.91 / F1 ≈ 0.92–0.95 are computed on the non-Uncertain subset only and are **misleading**.
- **Gemma 4 12B IT full-scale A1–A5** completed but is Reliable-heavy: Acc 0.814–0.825 ≈ always-Reliable prior 0.815; Spec 0.068–0.236. It went to full scale without the normal Stage 0 / balanced-100 / A1@1000 gates (documented in the canonical status doc).

Full tables: [`docs/FULL_SCALE_MODEL_COMPARISON.md`](docs/FULL_SCALE_MODEL_COMPARISON.md)

---

## Failure Modes

| Mode | What we observe | Example |
|------|-----------------|---------|
| **Single-class collapse** | One label on (nearly) all samples; Acc ≈ prior; Spec≈0; parser OK | LLaVA, Gemma @1000 |
| **Partial collapse (Reliable-heavy)** | Few Unreliable; Acc≈prior; Spec≈0 | MiniCPM A1@5747; Gemma 4 A1–A5@5747 |
| **Partial collapse (abstention-heavy)** | Uncertain dominates; binary metrics on a tiny scored subset | Molmo2 A1@5747; InternVL3.5 A1–A5@5747 |
| **Serialization / parse failure** | Non-empty model text that fails JSON parse | InternVL3 Stage 1 (13/100) |
| **Conservative abstention** | High Uncertain; binary metrics on a small scored subset | Molmo2; Qwen3 A5 |

We record **behavior**, not unproven causes (e.g. “ignored the image”).

---

## Current Experiments

No experiment jobs are running (checked 2026-09-27). The most recent full-scale runs are complete:

| Model | Experiment ID | Jobs (A1→A5) |
|-------|---------------|--------------|
| InternVL3.5-8B-HF | `20260924_internvl3_5_hf_A1A5_5747` | `8351993`–`8351997` |
| Gemma 4 12B IT | `20260925_gemma4_A1A5_5747` | `8353160`–`8353164` |

Qualification write-up: [`docs/INTERNVL3_5_HF_QUALIFICATION.md`](docs/INTERNVL3_5_HF_QUALIFICATION.md). Gate deviations: [`docs/EXPERIMENT_STATUS_CANONICAL.md`](docs/EXPERIMENT_STATUS_CANONICAL.md) §6.

---

## Repository Structure

```
configs/models/     Per-model YAML
scripts/            run_verification.py, submit_model_ablation.sh, eval
src/verification/   Runner, registry, records
src/lvm/            Model adapters + shared parser
jobs/               Slurm + model_runtime.sh
docs/               Canonical status, protocols, results
outputs/            Predictions + evaluation (authoritative evidence)
```

---

## Reproducing Experiments

```bash
# Dry-run (default)
./scripts/submit_model_ablation.sh qwen2_5_vl \
  --conditions A1,A2,A3,A4,A5 --limit 5747 --ablation-size 5747 \
  --experiment-id my_exp

# Submit
./scripts/submit_model_ablation.sh … --submit
```

Primary path: `scripts/submit_model_ablation.sh` → `jobs/run_verification.slurm`.  
New models: adapter + config + registry; preferred gate is Stage 0 → balanced-100 → A1@1000 → full A1–A5 only after gates pass. Historical models did not all follow this path (see canonical §6).

---

## Documentation

| Doc | Role |
|-----|------|
| [`docs/EXPERIMENT_STATUS_CANONICAL.md`](docs/EXPERIMENT_STATUS_CANONICAL.md) | **Source of truth** — inventory, jobs, collapse labels |
| [`docs/FULL_SCALE_MODEL_COMPARISON.md`](docs/FULL_SCALE_MODEL_COMPARISON.md) | Advisor A1–A5 @5747 tables |
| [`docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md`](docs/MODEL_SELECTION_AND_COLLAPSE_ANALYSIS.md) | Collapse forensics + next-model rationale |
| [`docs/INTERNVL3_5_HF_QUALIFICATION.md`](docs/INTERNVL3_5_HF_QUALIFICATION.md) | InternVL3 vs 3.5 controlled gate + InternVL3.5 full-scale outcome |
| [`docs/SUPPORTED_MODELS.md`](docs/SUPPORTED_MODELS.md) | Registry keys, adapters, configs per model |
| [`docs/INTERNVL3_QUALIFICATION.md`](docs/INTERNVL3_QUALIFICATION.md) | Original InternVL3 Stage 1 failure |
| [`docs/QWEN_FULL_A1_A5_RESULTS.md`](docs/QWEN_FULL_A1_A5_RESULTS.md) | Qwen2.5 detailed write-up |

---

## Author

**Annie Luo** · CS Honors Thesis · Mentor: **Fan Yang** · Wake Forest University · **2026**
