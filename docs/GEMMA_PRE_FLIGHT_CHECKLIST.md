# Gemma 3 Pre-Flight Checklist

Pre-cluster validation of the Gemma 3 12B IT integration (`--model gemma`).

Date: 2026-07-26  
Status: **Ready for cluster smoke testing** (pending checkpoint + environment verification on DEAC)

---

## Validation summary

| Area | Result | Notes |
|------|--------|-------|
| New Gemma files | **Pass** | Verifier, adapter, config, smoke scripts mirror LLaVA patterns |
| Registry | **Pass** | `gemma` registered; `gemma4` not registered; no accidental aliases |
| Existing models (qwen, llava) | **Pass** | Registry and adapter code paths unchanged |
| Frozen pipeline | **Pass** | No edits to evaluation, parser, prompts, visualization, runner, resume, output schema |
| Checkpoint config | **Pass** | `model_path` resolves correctly; existence not verified (cluster-only) |
| Transformers requirement | **Unknown** | Repo unpinned; verify on cluster before smoke |
| Infrastructure smoke scope | **Pass** | Load + inference only (no benchmark) |
| Framework smoke scope | **Pass** | 5 A1 samples via existing `run_verification.py` |
| Integration bugs found | **None** | No production code changes required |

---

## Phase 1 — File-by-file review

### New files (Gemma-only; no impact on existing models)

| File | Why added | Necessary? | Affects qwen/llava? |
|------|-----------|------------|---------------------|
| `src/lvm/gemma_verifier.py` | Gemma 3 model load + `generate_response()` using official `Gemma3ForConditionalGeneration` / `AutoProcessor` API | Yes | No — isolated module, loaded only for `--model gemma` |
| `src/lvm/gemma_verification_adapter.py` | Bridges frozen runner to Gemma verifier; identical structure to `LlavaVerificationAdapter` | Yes | No |
| `configs/models/gemma.yaml` | Checkpoint path and generation metadata for `--model gemma` | Yes | No |
| `scripts/smoke_tests/test_gemma3.py` | Standalone infrastructure smoke (CUDA + load + one inference) | Yes | No |
| `jobs/run_gemma_smoke_test.slurm` | Slurm wrapper for infrastructure smoke | Yes | No |
| `jobs/run_gemma_framework_smoke.slurm` | Slurm wrapper for 5-sample framework smoke | Yes | No |
| `docs/GEMMA_INTEGRATION_PLAN.md` | Pre-implementation design doc | Yes (documentation) | No |
| `docs/GEMMA_IMPLEMENTATION_REPORT.md` | Post-implementation summary | Yes (documentation) | No |

### Modified files (minimal registration only)

| File | Why modified | Necessary? | Affects qwen/llava? |
|------|--------------|------------|---------------------|
| `src/verification/registry.py` | Added `register_adapter("gemma", build_gemma_adapter)` | Yes | No — additive only; qwen/llava registrations unchanged |
| `src/config/model_config.py` | Added `"gemma": "gemma"` to `CANONICAL_MODEL_KEYS` | Yes | No — additive only; existing qwen/llava keys unchanged |

### Pre-existing (not introduced by Gemma 3 integration)

| Item | Notes |
|------|-------|
| `configs/models/gemma4.yaml` | Placeholder for future **Gemma 4**; predates Gemma 3 work |
| `"gemma4": "gemma4"` in `CANONICAL_MODEL_KEYS` | Path-resolution placeholder only; **not** registered in `registry.py` |
| `--model gemma4` | Correctly rejected at runtime (`Unknown verification model`) |

---

## Phase 2 — Registry validation

### Registered CLI keys

```
gemma, llava, qwen, qwen2_5_vl
```

### Alias map (unchanged)

| Input | Resolves to |
|-------|-------------|
| `qwen` | `qwen2_5_vl` |
| `qwen2_5_vl` | `qwen2_5_vl` |
| `llava` | `llava` |
| `gemma` | `gemma` |

**No Gemma aliases were added** (e.g. no `gemma3` → `gemma` mapping).

### Parity with qwen / llava

| Step | qwen | llava | gemma |
|------|------|-------|-------|
| CLI `--model` | ✓ | ✓ | ✓ |
| `create_adapter()` | ✓ | ✓ | ✓ |
| `resolve_model_config_path()` | ✓ | ✓ | ✓ |
| `resolve_model_checkpoint()` | ✓ | ✓ | ✓ |
| `VerificationRunner` orchestration | ✓ | ✓ | ✓ |
| `build_result_record()` output | ✓ | ✓ | ✓ |
| Shared parser | ✓ | ✓ | ✓ |

### gemma4 status

- **Not introduced** by this integration (framework-freeze placeholder).
- **Not registered** — do not remove; unrelated to Gemma 3.
- **Only supported Gemma registry key for verification:** `gemma`

---

## Phase 3 — Checkpoint validation

### `configs/models/gemma.yaml`

| Field | Value | Used by code? |
|-------|-------|---------------|
| `registry_key` | `gemma` | Documentation / consistency |
| `model_label` | `Gemma 3 12B IT` | `get_model_label()` |
| `model_id` | `null` | Skipped; allows `model_path` to win |
| `model_path` | `/deac/csc/yangGrp/luoz23/models/gemma-3-12b-it` | **`resolve_model_checkpoint("gemma")`** |
| `dtype` | `bfloat16` | Hardcoded in `GemmaVerifier` (same pattern as LLaVA) |
| `device_map` | `auto` | Passed via CLI default to adapter |
| `max_new_tokens` | `512` | CLI default in `run_verification.py` |
| `do_sample` | `false` | Hardcoded in verifier (`do_sample=False`) |
| `trust_remote_code` | `false` | Documentation only |
| `min_transformers` | `"4.50.0"` | Documentation only (not enforced in code) |

**Resolved checkpoint (local test):** `/deac/csc/yangGrp/luoz23/models/gemma-3-12b-it`

### Expected checkpoint directory

Location on DEAC:

```
/deac/csc/yangGrp/luoz23/models/gemma-3-12b-it/
```

Expected files (from official `google/gemma-3-12b-it` on Hugging Face):

```
config.json
generation_config.json
model.safetensors.index.json
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
preprocessor_config.json
processor_config.json
tokenizer.json
tokenizer.model
tokenizer_config.json
special_tokens_map.json
added_tokens.json
chat_template.json
```

Optional: `README.md`, `.gitattributes` (not required for inference).

**Do not assume the directory exists** — verify on cluster before smoke (see checklist below).

---

## Phase 4 — Transformers compatibility

### Repository requirements

| File | transformers pin |
|------|-------------------|
| `requirements.txt` | Unpinned (`transformers`) |
| `requirements_cluster.txt` | Unpinned (`transformers`) |

There is **no version pin in the repo**. Compatibility cannot be confirmed from source alone.

### Gemma 3 requirement

Official minimum: **transformers ≥ 4.50.0** (for `Gemma3ForConditionalGeneration`).

### Recommendation

**Do not upgrade Transformers preemptively.**

On the cluster, run the import check in the execution checklist (Step 3). Outcomes:

| Outcome | Action |
|---------|--------|
| Import succeeds, version ≥ 4.50 | **No upgrade** — proceed to smoke |
| Import fails | Upgrade only if needed: `pip install -U 'transformers>=4.50.0'` — then re-verify Qwen/LLaVA still import |
| Import succeeds but version < 4.50 | Upgrade required for Gemma only |

If Qwen and LLaVA smokes already pass on the cluster, record the installed version and compare to 4.50 before changing anything.

---

## Phase 5 — Infrastructure smoke validation

**Script:** `scripts/smoke_tests/test_gemma3.py`  
**Slurm:** `jobs/run_gemma_smoke_test.slurm`

### What it checks

| Check | Covered? |
|-------|----------|
| CUDA available | ✓ `check_cuda()` |
| Processor loading | ✓ `AutoProcessor.from_pretrained()` |
| Model loading | ✓ `Gemma3ForConditionalGeneration.from_pretrained()` |
| Chat template | ✓ `processor.apply_chat_template(..., tokenize=True)` |
| Image preprocessing | ✓ Image path in message content (HF-native) |
| Single-image inference | ✓ `model.generate()` |
| Decode | ✓ Trim input tokens + `processor.decode()` |

### What it does **not** check (by design — same as `test_llava_onevision.py`)

| Check | Where covered |
|-------|---------------|
| Parser | Framework smoke (`run_verification.py` + adapter) |
| JSON output schema | Framework smoke |
| `results_index.csv` | Framework smoke |
| Benchmark inference | Neither smoke (limit=5 only in framework smoke) |

**Does not run benchmark inference.** One descriptive prompt, 128 max tokens.

---

## Phase 6 — Framework smoke validation

**Slurm:** `jobs/run_gemma_framework_smoke.slurm`

### What it runs

```bash
python scripts/run_verification.py \
    --model gemma \
    --prompt-index <discovered A1_overlay_only/prompt_index.csv> \
    --results-dir outputs/verification/gemma/smoke_framework/A1 \
    --limit 5 \
    --batch-size 1 \
    --experiment-id smoke_framework \
    --condition A1
```

### What it does **not** modify

| Component | Modified? |
|-----------|-----------|
| Prompt templates | No — reads existing `.txt` files |
| Parser | No — uses `parse_verification_response` |
| Evaluation | No — not invoked |
| Output schema | No — uses `build_result_record` |
| Runner / resume | No — standard `VerificationRunner` |

### Post-run validation (embedded in Slurm job)

- `results_index.csv` exists
- ≥1 `sample_*.json` produced
- Fails only if **every** sample is `inference_error`
- Reports parse errors but does not fail on them (matches LLaVA framework smoke)

---

## Phase 7 — Risk assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Checkpoint directory missing or incomplete | **High** | Verify file list before smoke (Step 4) |
| Hugging Face Gemma license not accepted / no token | **High** | Accept license; ensure `HF_TOKEN` or `huggingface-cli login` on cluster |
| Transformers < 4.50 (import fails) | **High** | Import check (Step 3); upgrade only if import fails |
| GPU OOM (12B bf16 on L40S) | **Medium** | Framework smoke uses `batch_size=1`; 96G Slurm mem |
| JSON parse failures on verification prompts | **Medium** | Framework smoke reports; does not block infra smoke |
| Missing A1 ablation inputs (`prompt_index.csv`) | **Medium** | Framework smoke discovers under `outputs/`; fails clearly if missing |
| Processor / checkpoint mismatch | **Low** | Use official `google/gemma-3-12b-it` files only |
| Registry confusion (`gemma` vs `gemma4`) | **Low** | Only `gemma` is registered |
| Impact on Qwen / LLaVA | **Low** | Additive integration only; no shared code changed |

---

## Phase 8 — Remaining issues

| Issue | Blocker? |
|-------|----------|
| Gemma integration not yet committed to git on all machines | Yes — `git pull` after push |
| Checkpoint not verified on DEAC | Yes — before smoke |
| Transformers version not verified on DEAC | Yes — before smoke |
| Smoke tests not yet run on GPU | Expected — next step |
| `docs/SUPPORTED_MODELS.md` still lists Gemma 4 as planned | No — documentation drift only |

**No integration code fixes required** from this review.

---

## Cluster execution checklist

Run these commands **on DEAC** from the project root. Do not run benchmark or full A1.

### 1. Sync repository

```bash
cd /path/to/wild-palm-lvm-verification
git pull
```

### 2. Verify registry

```bash
python -c "
from src.verification.registry import get_registered_models, resolve_registry_key
from src.config.model_config import normalize_model_key, resolve_model_checkpoint
print('registered:', get_registered_models())
print('gemma ->', resolve_registry_key('gemma'), normalize_model_key('gemma'))
print('checkpoint:', resolve_model_checkpoint('gemma'))
"
python scripts/run_verification.py --help | grep -E 'model|gemma'
```

**Expected:** `registered: ('gemma', 'llava', 'qwen', 'qwen2_5_vl')`, checkpoint path printed, `--model {gemma,llava,qwen,qwen2_5_vl}`.

### 3. Verify Transformers / Gemma import

```bash
python -c "
import transformers
print('transformers:', transformers.__version__)
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
print('Gemma3 import: OK')
"
```

**Expected:** Version printed, `Gemma3 import: OK`. If `ImportError`, resolve before smoke (see Phase 4).

### 4. Verify checkpoint exists

```bash
CKPT=/deac/csc/yangGrp/luoz23/models/gemma-3-12b-it
test -d "$CKPT" && echo "directory OK" || echo "MISSING: $CKPT"
ls -1 "$CKPT"/config.json "$CKPT"/model.safetensors.index.json "$CKPT"/processor_config.json "$CKPT"/tokenizer.json 2>&1
ls -1 "$CKPT"/model-*-of-*.safetensors 2>&1 | wc -l
```

**Expected:** `directory OK`, config/processor/tokenizer files present, **5** shard files.

### 5. Run infrastructure smoke

```bash
sbatch jobs/run_gemma_smoke_test.slurm
```

### 6. Inspect infrastructure smoke log

```bash
# Replace JOBID with the id from sbatch output
tail -f logs/slurm/gemma_smoke_JOBID.out
```

**Expected successful output:**

```
CUDA device: NVIDIA L40S
Loading processor...
Loading model...
Running inference...
Inference time: ...s
Generated response:
<non-empty text>
Smoke test PASSED
```

### 7. Run framework smoke

```bash
sbatch jobs/run_gemma_framework_smoke.slurm
```

### 8. Inspect framework smoke log

```bash
tail -f logs/slurm/gemma_framework_smoke_JOBID.out
```

**Expected successful output:**

```
Running framework smoke inference...
Gemma 3 verification inference
  Model key:    gemma
  Samples:      5
...
=== Framework smoke validation ===
results_index.csv exists: True
sample_*.json count: 5
...
Framework smoke validation PASSED
```

**Expected artifacts:**

```
outputs/verification/gemma/smoke_framework/A1/results_index.csv
outputs/verification/gemma/smoke_framework/A1/sample_*.json   (5 files)
```

Each JSON should include: `sample_id`, `raw_response`, `decision`, `model_key` (`gemma`), `timestamp`, and empty `inference_error` on success.

---

## Expected successful outputs (summary)

| Stage | Success criterion |
|-------|-------------------|
| Registry | `gemma` in registered models; checkpoint path resolves |
| Transformers | `Gemma3ForConditionalGeneration` imports |
| Checkpoint | Directory + 5 shards + processor/tokenizer configs |
| Infrastructure smoke | `Smoke test PASSED` in Slurm log |
| Framework smoke | 5 JSON files + `results_index.csv`; not all `inference_error` |

---

See also: [`GEMMA_INTEGRATION_PLAN.md`](GEMMA_INTEGRATION_PLAN.md) · [`GEMMA_IMPLEMENTATION_REPORT.md`](GEMMA_IMPLEMENTATION_REPORT.md)
