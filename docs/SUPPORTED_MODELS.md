# Supported Models

Implementation status of VLM backends in the palm verification framework (updated 2026-09-27).

This file lists **what is integrated** (adapter + config + registry). Run status and scientific outcome live in [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md); numeric results live in [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md).

Adding a new model requires only: verifier + adapter + config YAML + registry entry (+ optional runtime case in `jobs/lib/model_runtime.sh`). See [`FRAMEWORK_FREEZE.md`](FRAMEWORK_FREEZE.md). The shared prompts, parser, evaluator and GT matching are frozen and are not model-specific.

---

## Registry summary

All keys below are registered in `src/verification/registry.py` and resolved by `src/config/model_config.py` / `jobs/lib/model_runtime.sh`.

| Registry key | Aliases | Checkpoint (local, `/deac/csc/yangGrp/luoz23/models/…`) | Verifier / adapter (`src/lvm/`) | Config | `trust_remote_code` | Min transformers | Run status |
|--------------|---------|-------------------------------------------------------|----------------------------------|--------|:---:|------------------|-----------|
| `qwen2_5_vl` | `qwen`, `qwen25_vl_7b`, `qwen2_5_vl_7b` | `Qwen2.5-VL-7B-Instruct` | `qwen_verifier.py` / `qwen_verification_adapter.py` | `configs/models/qwen2_5_vl.yaml` | no | cluster pin (`requirements_cluster.txt`) | Complete |
| `qwen3_vl` | `qwen3_vl_8b`, `qwen3vl`, `qwen3vl_8b` | `Qwen3-VL-8B-Instruct` | `qwen3_vl_verifier.py` / `qwen3_vl_verification_adapter.py` | `configs/models/qwen3_vl.yaml` | no | 4.57.0 | Complete |
| `glm_4_6v_flash` | `glm46v_flash` | `GLM-4.6V-Flash` | `glm_4_6v_flash_verifier.py` / `glm_4_6v_flash_verification_adapter.py` | `configs/models/glm_4_6v_flash.yaml` | no | 4.57.0 (runs pinned 5.17.0) | Complete |
| `phi4_multimodal` | `phi4` | `Phi-4-multimodal-instruct` | `phi4_multimodal_verifier.py` / `phi4_multimodal_verification_adapter.py` | `configs/models/phi4_multimodal.yaml` | yes | 4.48.2 | Complete |
| `internvl3_5_hf` | `internvl35_hf`, `internvl3.5_hf` | `InternVL3_5-8B-HF` (`OpenGVLab/InternVL3_5-8B-HF`) | `internvl3_5_hf_verifier.py` / `internvl3_5_hf_verification_adapter.py` | `configs/models/internvl3_5_hf.yaml` | no | 4.52.1 | Complete |
| `internvl3` | `internvl`, `internvl3_8b` | `InternVL3-8B-Instruct` | `internvl_verifier.py` / `internvl_verification_adapter.py` | `configs/models/internvl3.yaml` | yes | 4.37.2 | Qualification failed / stopped |
| `minicpm_v4_5` | `minicpm45`, `minicpm-v-4.5`, `minicpm_v45` | `MiniCPM-V-4_5` | `minicpm_v4_5_verifier.py` / `minicpm_v4_5_verification_adapter.py` | `configs/models/minicpm_v4_5.yaml` | yes | 4.51.0 (pinned) | A1 only |
| `molmo2_8b` | `molmo2`, `molmo2-8b` | `Molmo2-8B` | `molmo2_verifier.py` / `molmo2_verification_adapter.py` | `configs/models/molmo2_8b.yaml` | yes | 4.57.1 (pinned) | A1 only |
| `llava` | — | `llava_onevision` | `llava_verifier.py` / `llava_verification_adapter.py` | `configs/models/llava.yaml` | no | — | A1 only (@1000) |
| `gemma` (Gemma 3) | — | `gemma-3-12b-it` | `gemma_verifier.py` / `gemma_verification_adapter.py` | `configs/models/gemma.yaml` | no | 4.50.0 | A1 only (@1000) |
| `gemma4` (Gemma 4 12B IT) | `gemma4_12b`, `gemma-4`, `gemma4_12b_it` | `gemma-4-12B-it` (`google/gemma-4-12B-it`) | `gemma4_verifier.py` / `gemma4_verification_adapter.py` | `configs/models/gemma4.yaml` | no | 5.10.1 (runs used 5.17.0 in `envs/wild-palm-gemma4`) | Complete |
| `llama3_2_11b_vision` | — | `Llama-3.2-11B-Vision-Instruct` (`meta-llama/Llama-3.2-11B-Vision-Instruct`) | `llama3_2_11b_vision_verifier.py` / `llama3_2_11b_vision_verification_adapter.py` | `configs/models/llama3_2_11b_vision.yaml` | no | 4.45.0 | Integrated; engineering qualification only |
| `ministral3_8b` | — | `Ministral-3-8B-Instruct-2512-BF16` | `ministral3_8b_verifier.py` / `ministral3_8b_verification_adapter.py` | `configs/models/ministral3_8b.yaml` | no | 5.0.0 | Integrated; not evaluated |

---

## Model-specific notes

- **Qwen2.5-VL** — production experiments predate the per-model output namespace and live under the legacy key `outputs/verification/qwen/`. Path helpers read these automatically.
- **Phi-4** — the verifier explicitly sets `<|end|>` / EOS stop ids and logs `PHI4_GEN_TOKENS` per sample; this fix produced the completed A2@1000 re-run and all A1–A5@5747 runs. Jobs refuse GPUs with uncorrected ECC errors.
- **InternVL3.5-8B-HF** — HF-native `AutoProcessor` + `AutoModelForImageTextToText`; ran on the default cluster Python (3.9.25, transformers 4.57.6). Launched via `jobs/run_internvl3_5_hf_Ax_5747.slurm`.
- **Gemma 4 12B IT** — `AutoModelForMultimodalLM` with `enable_thinking=False`; requires the isolated venv built by `jobs/setup_gemma4_env.slurm` (`GEMMA4_VENV`). Sample JSONs from the completed run have no `generation` metadata key.
- **MiniCPM-V-4.5 / Molmo2-8B** — full-scale A1 ran on H200 (`gpu_small`); isolated venvs with exact transformers pins asserted at job start.
- **Llama-3.2-11B-Vision-Instruct** — `MllamaForConditionalGeneration` + `MllamaProcessor`. Only an engineering Stage 0 run exists (`jobs/run_llama3_2_11b_vision_engqual.slurm`); the scientific gate (`jobs/run_llama3_2_11b_vision_qual.slurm`) has not been run.
- **Ministral-3-8B** — `Mistral3ForConditionalGeneration` (transformers ≥ 5.0). Download and qualification jobs exist (`jobs/download_ministral3_8b.slurm`, `jobs/run_ministral3_8b_qual.slurm`); no experiment has been run.

---

## Output paths

```
outputs/verification/<registry_key>/<experiment_id>/<A1..A5>/
outputs/evaluation_protocol_v2/<registry_key>/<experiment_id>/<A1..A5>/   # current (Protocol v2)
outputs/evaluation/<registry_key>/<experiment_id>/<A1..A5>/               # Protocol v1, frozen
```

All trees are gitignored; they are on-disk evidence only. Numeric results: [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md).

---

See also: [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`archive/docs/MULTI_MODEL_INTEGRATION_PLAN.md`](../archive/docs/MULTI_MODEL_INTEGRATION_PLAN.md) (historical July 2026 design; its "planned" statuses are superseded)
