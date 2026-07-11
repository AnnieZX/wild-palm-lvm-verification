# Architectural contract for the verification framework (frozen July 2026)

This document defines the **frozen interfaces** for the wild palm VLM verification pipeline.

System diagrams: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)

After this freeze, adding a new model requires only:

- a new verifier (`src/lvm/<model>_verifier.py`)
- a new adapter (`src/lvm/<model>_verification_adapter.py`)
- a model config (`configs/models/<registry_key>.yaml`)
- a registry entry in `src/verification/registry.py`

The core framework under `src/verification/` must **not** change for new models.

---

## Frozen APIs

### VerificationRunner — `src/verification/runner.py`

| | |
|--|--|
| **Method** | `run(jobs, config: RunnerConfig) -> pd.DataFrame` |
| **Ownership** | CLI entry points (`scripts/run_verification.py`) |
| **Responsibility** | Resume filtering, job iteration, output persistence |
| **Models** | Must not bypass; adapters implement `verify()` only |

### VerificationJob — `src/verification/jobs.py`

| Field | Type |
|-------|------|
| `sample_id` | `str` |
| `image_path` | `Path` |
| `prompt_path` | `Path` |

Loaders: `load_verification_jobs()`, `validate_jobs()`.

### BaseVerificationAdapter — `src/verification/base_adapter.py`

| | |
|--|--|
| **Property** | `model_label: str` |
| **Method** | `verify(job) -> VerificationOutcome` |
| **Outcome** | `record: dict`, `status: str` |

Status values: `ok`, `parse_error`, `inference_error`.

### build_result_record() — `src/verification/records.py`

Canonical JSON schema builder. All adapters must call this.

### VerificationOutputManager — `src/verification/output_manager.py`

| Method | Role |
|--------|------|
| `save_json(sample_id, record)` | Write `sample_*.json` |
| `finalize_index(new_rows, resume=)` | Write `results_index.csv` |

### Registry — `src/verification/registry.py`

| Function | Role |
|----------|------|
| `register_adapter(key, factory)` | Register adapter factory |
| `create_adapter(model, **kwargs)` | Instantiate adapter |
| `get_registered_models()` | List valid `--model` choices |

### Prompt builder — `src/prompts/ablation_verification_prompts.py`

| Function | Role |
|----------|------|
| `build_ablation_verification_prompt(metadata, condition)` | Build A1–A5 prompt text |

Prompts are **pre-generated** to disk. Adapters read `.txt` files; they do not rebuild prompts.

### Response parser — `src/lvm/parsers/base.py`

Pipeline: `normalize_raw_response()` → `parse_json_response()` → `normalize_decision()`.

Shim: `src/lvm/verification_response_parser.py` re-exports for backward compatibility.

### Evaluation API — `scripts/evaluate_verification_against_groundtruth.py`

Reads `decision` from `sample_*.json`. Writes `{code}_evaluation.csv`.

IoU threshold: **0.5**. Greedy one-to-one matching via `src/evaluation/gt_matching.py`.

### Metrics API — `scripts/compute_verification_metrics.py`

Reads `*_evaluation.csv`. Writes `{code}_metrics.json` and `summary.csv`.

Excludes **Uncertain** from binary Precision/Recall/F1.

---

## Frozen output schema

Every `sample_*.json` must contain at minimum:

| Field | Required | Evaluation uses? |
|-------|----------|------------------|
| `sample_id` | Yes | Join key |
| `decision` | Yes (may be empty on error) | **Yes** |
| `confidence_reasoning` | Yes (may be empty) | No (visualization) |
| `visual_reasoning` | Yes (may be empty) | No (visualization) |
| `raw_response` | Yes | No (audit) |
| `parsed_response` | Yes (`null` on failure) | No |
| `runtime_seconds` | Yes | No |
| `parse_error` | Yes (empty if OK) | No |
| `inference_error` | Yes (empty if OK) | No |
| `timestamp` | Yes (ISO-8601 UTC) | No |

Optional audit metadata (added at freeze, do not remove):

| Field | Purpose |
|-------|---------|
| `model_key` | Registry key (e.g. `qwen2_5_vl`) |
| `model_name` | Checkpoint path or Hugging Face id |
| `condition` | Ablation code when applicable |
| `experiment_id` | Experiment timestamp id |

**Evaluation scripts ignore all fields except `decision`.** Existing Qwen2.5 results without metadata remain valid.

### Allowed decision labels

`Reliable`, `Uncertain`, `Unreliable` — only these three.

Invalid or unparseable outputs must **not** be silently coerced. Use `parse_error` and empty `decision`.

---

## Frozen prompt semantics

Defined in `src/prompts/ablation_verification_prompts.py` and `docs/ABLATION_STUDY.md`.

All models receive identical pre-built prompt `.txt` files per A1–A5 condition. Model-specific chat templates may wrap the text, but semantic task content is fixed.

JSON response template (frozen):

```json
{
  "decision": "Reliable | Uncertain | Unreliable",
  "confidence_reasoning": "",
  "visual_reasoning": ""
}
```

---

## Frozen evaluation

Protocol: `docs/EVALUATION_PROTOCOL.md`

- GT from LabelMe `label == "palm"`
- Greedy IoU matching, threshold 0.5
- Matched detection = GT positive; unmatched = GT negative
- Uncertain predictions excluded from binary metrics

---

## Frozen metrics

From `scripts/compute_verification_metrics.py`:

- Positive prediction: `Reliable`
- Negative prediction: `Unreliable`
- Binary TP/FP/FN/TN from `matched_gt` × `decision`
- Report Uncertain count separately

---

## Frozen resume

Resume key: **`sample_id` within a results directory**.

Logical identity: `(model_key, condition, sample_id)` implemented via directory isolation:

```
outputs/verification/<model_key>/<experiment_id>/<condition>/sample_*.json
outputs/verification/<model_key>/<experiment_id>/<condition>/results_index.csv
```

Documented in `src/utils/verification_resume.py`.

Legacy Qwen experiments under `outputs/verification/qwen/` are detected automatically when resuming `qwen2_5_vl` runs.

---

## Frozen visualization inputs

`scripts/visualization/visualize_verification.py` reads:

- Verification JSON from `outputs/verification/<model_key>/<experiment_id>/A*`
- Evaluation CSV from `outputs/evaluation/<model_key>/<experiment_id>/A*`
- Shared dataset overlays and LabelMe GT

---

## Registry naming (frozen)

| Registry key | Status | Notes |
|--------------|--------|-------|
| `qwen2_5_vl` | **Primary** | Qwen2.5-VL production baseline |
| `qwen` | **Alias** | Maps to same adapter as `qwen2_5_vl` |
| `llava` | Planned | Not yet implemented |
| `gemma4` | Planned | Not yet implemented |
| `qwen3_vl` | Planned | Not yet implemented |

Output directory name = canonical registry key (`normalize_model_key()`).

---

## Cross-Model Fairness Contract

### Fixed across all models (must not vary)

| Component | Location |
|-----------|----------|
| Verification dataset | `outputs/verification_dataset/` |
| Sample IDs | `sample_XXXXXX` |
| A1–A5 definitions | `ablation_verification_prompts.py` |
| Ablation image files | Shared PNG inputs per condition |
| Prompt semantic content | Pre-built `.txt` files |
| Output label definitions | Reliable / Uncertain / Unreliable |
| Output JSON schema | `build_result_record()` |
| GT matching | IoU 0.5 greedy |
| Metrics formulas | `compute_verification_metrics.py` |
| Uncertain exclusion | Evaluation protocol |
| Resume convention | Per-directory `sample_id` |
| Failed output treatment | `parse_error` / empty decision |

### Variable across models (document in experiment metadata)

| Component | Notes |
|-----------|-------|
| Model weights | Checkpoint path in config YAML |
| Vision encoder | Architecture-specific |
| Processor / tokenizer | Model-specific |
| Chat template | Wrapping of shared prompt text |
| Generation parameters | `do_sample`, dtype, etc. |
| Raw response format | Normalized before shared parser |

---

## Output directory layout (frozen)

```
outputs/
  verification/
    qwen2_5_vl/<experiment_id>/A1/sample_*.json
    llava/<experiment_id>/A1/          # future
  evaluation/
    qwen2_5_vl/<experiment_id>/A1/A1_evaluation.csv
  visualization/
    qwen2_5_vl/<experiment_id>/overlay/
```

Legacy: `outputs/verification/qwen/` (pre-freeze experiments) remains readable.

---

## Configuration (frozen pattern)

Per-model YAML: `configs/models/<registry_key>.yaml`

Resolution order for checkpoint path:

1. CLI `--model-path`
2. Config keys: `model_id`, `model_path`, `active_model`
3. Legacy `configs/model.yaml` fallback for `qwen2_5_vl` only

---

*Framework frozen July 2026. See `docs/SUPPORTED_MODELS.md` for per-model status.*
