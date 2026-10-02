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

- GT from LabelMe shapes where `is_palm_label(label)` holds, i.e. `isinstance(label, str) and label.strip().lower() == "palm"` (Protocol v2; see [Amendment 2026-09-27](#amendment-2026-09-27--evaluation-protocol-v2))
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
- Evaluation CSV from `outputs/evaluation_protocol_v2/<model_key>/<experiment_id>/A*` (current; Protocol v1 CSVs remain in `outputs/evaluation/`)
- Shared dataset overlays and LabelMe GT

---

## Registry naming (frozen)

Canonical keys registered in `src/verification/registry.py` (aliases such as `qwen`, `phi4`, `gemma4_12b` map to them):

`qwen2_5_vl` · `qwen3_vl` · `llava` · `gemma` · `gemma4` · `internvl3` · `internvl3_5_hf` · `glm_4_6v_flash` · `phi4_multimodal` · `molmo2_8b` · `minicpm_v4_5` · `llama3_2_11b_vision` · `ministral3_8b`

Per-model status: [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) and [`EXPERIMENT_STATUS_CANONICAL.md`](EXPERIMENT_STATUS_CANONICAL.md).

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
    llava/<experiment_id>/A1/
  evaluation_protocol_v2/              # current evaluation (Protocol v2)
    qwen2_5_vl/<experiment_id>/A1/A1_evaluation.csv
  evaluation/                          # Protocol v1, frozen provenance
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

## Amendment 2026-09-27 — Evaluation Protocol v2

**Change.** The GT palm-label rule changed from the case-sensitive `label == "palm"` to `isinstance(label, str) and label.strip().lower() == "palm"`, implemented once as `is_palm_label()` in `src/preprocessing/gt_palm_bboxes.py` (`EVALUATION_PROTOCOL_VERSION = "v2"`).

**Reason.** 70 LabelMe files (parents 0194–0205) label palms `"Palm"`; v1 silently dropped their 486 boxes, mislabeling 424 of the 5,747 verification detections as GT−.

**Unchanged (still frozen).** Verification dataset, sample IDs, A1–A5 inputs and prompts, parser, output schema, decision labels, IoU threshold 0.5, greedy one-to-one matching, Uncertain exclusion, resume convention, and all stored predictions. No inference was rerun.

**Additive.** `A*_metrics.json` gains `specificity`, `balanced_accuracy` and `evaluation_protocol`. Evaluation outputs are versioned: v2 in `outputs/evaluation_protocol_v2/` (with `PROTOCOL.json`), v1 frozen in `outputs/evaluation/`. `jobs/run_verification.slurm` refuses to write into the v1 tree.

Details: [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) · [`EXPERIMENT_RESULTS_CANONICAL.md`](EXPERIMENT_RESULTS_CANONICAL.md) §2.

---

## Amendment 2026-10-01 — Ministral 3 raw-response normalizer

**Change.** `normalize_raw_response()` (`src/lvm/parsers/cleanup.py`) gains its first
model-specific normalizer, for `ministral3_3b`, `ministral3_8b` and `ministral3_14b` only:
`escape_control_chars_in_json_strings()` escapes literal newline / carriage-return / tab
(and other < 0x20) characters that occur **inside JSON string literals**.

**Reason.** Ministral 3 returns otherwise valid response objects whose `visual_reasoning`
value starts and ends with literal newlines; strict `json.loads` rejects them as invalid
control characters (7/10 failures on the Stage 0 set, all of them Unreliable responses).

**Scope.** Equivalent to what `json.loads(..., strict=False)` accepts; decoded string
content is unchanged and nothing outside string literals is touched. No decision is
inferred or coerced: invalid labels, truncated objects and free text still fail. Every other
model key remains an identity pass-through. `parse_json_response()`, `normalize_decision()`,
the label set and the output schema are unchanged. No stored prediction is re-parsed.
Records written with it carry `generation.raw_response_normalizer`. Tests:
`tests/test_model_expansion_integration.py::TestMinistral3Normalizer`.

---

*Framework frozen July 2026; amended 2026-09-27 (Protocol v2) and 2026-10-01 (Ministral 3 normalizer). See `docs/SUPPORTED_MODELS.md` for per-model status.*
