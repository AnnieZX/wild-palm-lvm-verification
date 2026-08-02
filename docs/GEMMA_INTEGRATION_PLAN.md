# Gemma 3 12B IT Integration Plan

Target checkpoint: **`google/gemma-3-12b-it`**

Registry key: **`gemma`** (`--model gemma`)

This plan integrates Gemma 3 as a third verification backend without changing the frozen pipeline (prompts, parser, evaluation, visualization, resume, or output schema).

---

## 1. Architecture comparison

| Stage | Qwen2.5-VL | LLaVA-OneVision | Gemma 3 12B IT |
|-------|------------|-----------------|----------------|
| **Model class** | `Qwen2_5_VLForConditionalGeneration` | `LlavaOnevisionForConditionalGeneration` | `Gemma3ForConditionalGeneration` |
| **Processor** | `AutoProcessor` | `LlavaOnevisionProcessor` | `AutoProcessor` |
| **Extra vision utils** | `qwen_vl_utils.process_vision_info` | None | None (images in chat template) |
| **Message format** | `[{role, content: [{type:image, image:path}, {type:text, text}]}]` | Same structure | Same structure |
| **Chat template** | `apply_chat_template(..., tokenize=False)` then separate `processor()` | `apply_chat_template(..., add_generation_prompt=True)` then `processor(images, text)` | `apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)` |
| **Image preprocessing** | Via `process_vision_info` + processor | `PIL.Image.open().convert("RGB")` passed to processor | Embedded in `apply_chat_template` from message `image` field |
| **Device / dtype** | `inputs.to(model.device)`, `torch_dtype="auto"` | `bfloat16`, `_move_inputs_to_model` | `bfloat16`, `inputs.to(model.device, dtype=bfloat16)` |
| **Generate** | `model.generate(**inputs, max_new_tokens=512, do_sample=False)` | Same | Same |
| **Decode** | Trim input token ids, `processor.batch_decode` | Trim by input length, `processor.decode` | Trim by input length, `processor.decode` |
| **Adapter pattern** | `QwenVerificationAdapter.verify()` → verifier → `parse_verification_response` → `build_result_record` | Identical orchestration | Identical orchestration |
| **Batch inference** | `generate_batch_responses()` in verifier | Single-sample only (runner batch_size=1 typical) | Single-sample only (12B memory) |
| **Min Transformers** | Qwen2.5-VL pin | ≥4.45 (LLaVA-OneVision) | **≥4.50.0** (Gemma 3) |

**Pipeline philosophy alignment:** Gemma follows the **LLaVA single-path** pattern (adapter reads prompt file → verifier runs one image+text generation → shared parser → shared JSON record). Unlike Qwen, Gemma tokenizes inside `apply_chat_template` and does not require a separate vision-info helper.

---

## 2. Required new files

| File | Purpose |
|------|---------|
| `src/lvm/gemma_verifier.py` | Load `Gemma3ForConditionalGeneration` + `AutoProcessor`; `generate_response()` |
| `src/lvm/gemma_verification_adapter.py` | `GemmaVerificationAdapter` + `build_gemma_adapter` factory |
| `configs/models/gemma.yaml` | Checkpoint path, dtype, generation defaults |
| `scripts/smoke_tests/test_gemma3.py` | Standalone CUDA + load + single-image inference |
| `jobs/run_gemma_smoke_test.slurm` | Cluster smoke (infrastructure) |
| `jobs/run_gemma_framework_smoke.slurm` | 5-sample framework smoke via `run_verification.py --model gemma` |
| `docs/GEMMA_INTEGRATION_PLAN.md` | This document |
| `docs/GEMMA_IMPLEMENTATION_REPORT.md` | Post-implementation summary |

---

## 3. Required edits

| File | Change |
|------|--------|
| `src/verification/registry.py` | `register_adapter("gemma", build_gemma_adapter)` |
| `src/config/model_config.py` | Add `"gemma": "gemma"` to `CANONICAL_MODEL_KEYS` |

**No edits** to: parser, prompts, evaluation scripts, visualization, runner, output manager, or existing Qwen/LLaVA adapters.

---

## 4. Expected implementation difficulty

| Area | Difficulty | Notes |
|------|------------|-------|
| Verifier + adapter | **Low–Medium** | Mirrors LLaVA; official HF API is well documented |
| Registry / config | **Low** | Two-line registration + YAML |
| Transformers version | **Medium** | Cluster pin may need `transformers>=4.50.0` for Gemma 3 |
| Parser compatibility | **Low** | Shared parser unchanged; JSON decision format identical |
| Hugging Face gating | **Low–Medium** | Gemma requires HF account acceptance + token on cluster |

Overall: **Medium** — mostly boilerplate following LLaVA, with version and memory risk on cluster.

---

## 5. Expected GPU memory

| Component | Estimate (bf16, L40S 48 GB) |
|-----------|----------------------------|
| Weights (12B) | ~24 GB |
| Vision encoder + KV cache (512 tokens, batch=1) | ~4–8 GB |
| Activations / overhead | ~2–4 GB |
| **Total** | **~30–36 GB** |

Fits a single **L40S (48 GB)** at `batch_size=1`. Full benchmark should use batch_size=1 for Gemma 12B (same as LLaVA smoke defaults).

---

## 6. Expected runtime on one L40S

| Stage | Estimate |
|-------|----------|
| Model load (first time) | 2–5 min |
| Single A1 sample (896×896 vision tokens, 512 max_new_tokens) | 10–20 s |
| 5-sample framework smoke | 1–3 min inference + load |
| 1000-sample A1 (extrapolation) | ~3–6 h at batch_size=1 |

Slower than LLaVA 7B (~2× parameter count); comparable to or slightly faster than Qwen2.5-VL-7B depending on vision token count.

---

## 7. Potential compatibility risks

| Risk | Mitigation |
|------|------------|
| **Transformers < 4.50** | Smoke test checks import of `Gemma3ForConditionalGeneration`; document upgrade in implementation report |
| **HF Gemma license gating** | Use local `model_path` on cluster after manual download; clear error if auth fails |
| **OOM on 48 GB** | Default `batch_size=1`; bf16 + `device_map=auto`; no batch path in verifier |
| **High-res / non-square images** | Gemma supports `do_pan_and_scan=True` for artifacts; default **False** for parity with LLaVA/Qwen smoke (enable later if needed) |
| **System message drift** | Do **not** inject a system prompt; user message only (same as LLaVA verification path) |
| **JSON parse failures** | Handled by existing parser; smoke test reports parse_error without failing job |
| **Confusion with planned `gemma4`** | Separate registry key `gemma` vs placeholder `gemma4`; no changes to gemma4.yaml |

---

## Implementation checklist

- [ ] `gemma_verifier.py` — load + `generate_response()`
- [ ] `gemma_verification_adapter.py` — `verify()` identical to LLaVA adapter
- [ ] `configs/models/gemma.yaml`
- [ ] Registry + `CANONICAL_MODEL_KEYS`
- [ ] `test_gemma3.py` + Slurm jobs
- [ ] Framework smoke: 5 samples, validate `sample_*.json` + `results_index.csv`
- [ ] `GEMMA_IMPLEMENTATION_REPORT.md`

---

See also: [`docs/FRAMEWORK_FREEZE.md`](FRAMEWORK_FREEZE.md) · [`docs/SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md)
