# Supported Models

Status of VLM backends for the palm verification framework.

Adding a new model requires only: verifier + adapter + config YAML + registry entry. See `docs/FRAMEWORK_FREEZE.md`.

---

## Qwen2.5-VL

| Item | Value |
|------|-------|
| **Registry key** | `qwen2_5_vl` (alias: `qwen`) |
| **Config** | `configs/models/qwen2_5_vl.yaml` |
| **Legacy config** | `configs/model.yaml` (`active_model` fallback) |
| **Adapter** | `src/lvm/qwen_verification_adapter.py` |
| **Verifier** | `src/lvm/qwen_verifier.py` |
| **Min Transformers** | Qwen2.5-VL support (cluster pin in `requirements_cluster.txt`) |
| **Extra deps** | `qwen-vl-utils` |
| **Status** | **Production — active baseline** |
| **Phase** | Complete |

```bash
python scripts/run_verification.py \
  --model qwen2_5_vl \
  --prompt-index outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv \
  --results-dir outputs/verification/qwen2_5_vl/my_run/A1
```

---

## LLaVA

| Item | Value |
|------|-------|
| **Registry key** | `llava` |
| **Config** | `configs/models/llava.yaml` |
| **Adapter** | `src/lvm/llava_verification_adapter.py` *(planned)* |
| **Verifier** | `src/lvm/llava_verifier.py` *(planned)* |
| **Min Transformers** | LLaVA-NeXT support (≥4.48 recommended) |
| **Proposed checkpoint** | `llava-hf/llava-v1.6-mistral-7b-hf` |
| **Status** | **Planned** |
| **Phase** | Phase 1 (first architecture-diverse backend) |

---

## Gemma 4

| Item | Value |
|------|-------|
| **Registry key** | `gemma4` |
| **Config** | `configs/models/gemma4.yaml` |
| **Adapter** | `src/lvm/gemma4_verification_adapter.py` *(planned)* |
| **Verifier** | `src/lvm/gemma4_verifier.py` *(planned)* |
| **Min Transformers** | Gemma 4 support (likely newer than cluster pin — verify at implementation) |
| **Proposed checkpoint** | `google/gemma-4-E4B-it` |
| **Status** | **Planned** |
| **Phase** | Phase 2 |

---

## Qwen3-VL

| Item | Value |
|------|-------|
| **Registry key** | `qwen3_vl` |
| **Config** | `configs/models/qwen3_vl.yaml` |
| **Adapter** | `src/lvm/qwen3_vl_verification_adapter.py` *(planned)* |
| **Verifier** | `src/lvm/qwen3_vl_verifier.py` *(planned)* |
| **Min Transformers** | Qwen3-VL support (≥4.57 or install from source — verify at implementation) |
| **Proposed checkpoint** | `Qwen/Qwen3-VL-8B-Instruct` |
| **Status** | **Planned** |
| **Phase** | Phase 3 (generational comparison vs Qwen2.5-VL) |

---

## Registry summary

| Key | Alias | Adapter registered | Implemented |
|-----|-------|-------------------|-------------|
| `qwen2_5_vl` | — | Yes | Yes |
| `qwen` | → `qwen2_5_vl` | Yes | Yes |
| `llava` | — | No | No |
| `gemma4` | — | No | No |
| `qwen3_vl` | — | No | No |

---

## Output paths

All models write to:

```
outputs/verification/<registry_key>/<experiment_id>/<A1..A5>/
outputs/evaluation/<registry_key>/<experiment_id>/<A1..A5>/
```

Legacy Qwen2.5 experiments may exist under `outputs/verification/qwen/` (pre-freeze). The path helpers detect and read these automatically.

---

See also: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/MULTI_MODEL_INTEGRATION_PLAN.md`](docs/MULTI_MODEL_INTEGRATION_PLAN.md)
