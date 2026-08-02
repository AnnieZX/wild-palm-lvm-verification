# Gemma 3 12B IT Implementation Report

Integration of **`google/gemma-3-12b-it`** as registry key **`gemma`** into the frozen verification framework.

Date: 2026-07-26

---

## Summary

Gemma 3 is integrated as a third model backend following the same adapter → verifier → parser → JSON → `results_index.csv` path used by Qwen2.5-VL and LLaVA-OneVision. No changes were made to the evaluation pipeline, parser, prompts, visualization, runner, or output schema.

---

## Files added

| File | Purpose |
|------|---------|
| `src/lvm/gemma_verifier.py` | `GemmaVerifier` — load + single-sample `generate_response()` |
| `src/lvm/gemma_verification_adapter.py` | `GemmaVerificationAdapter` + `build_gemma_adapter` |
| `configs/models/gemma.yaml` | Checkpoint, dtype, generation defaults |
| `scripts/smoke_tests/test_gemma3.py` | Standalone infrastructure smoke test |
| `jobs/run_gemma_smoke_test.slurm` | Cluster Slurm job for infrastructure smoke |
| `jobs/run_gemma_framework_smoke.slurm` | 5-sample framework smoke via `run_verification.py` |
| `docs/GEMMA_INTEGRATION_PLAN.md` | Pre-implementation architecture plan |
| `docs/GEMMA_IMPLEMENTATION_REPORT.md` | This report |

---

## Files modified

| File | Change |
|------|--------|
| `src/verification/registry.py` | Registered `build_gemma_adapter` under key `gemma` |
| `src/config/model_config.py` | Added `"gemma": "gemma"` to `CANONICAL_MODEL_KEYS` |

---

## Registration

```bash
python scripts/run_verification.py --model gemma ...
```

Registered keys (unchanged for existing models):

| Key | Status |
|-----|--------|
| `qwen2_5_vl` / `qwen` | Unchanged |
| `llava` | Unchanged |
| `gemma` | **New** |

Local verification:

```
registered: ('gemma', 'llava', 'qwen', 'qwen2_5_vl')
```

---

## Smoke test result

**Not executed in this session** — requires DEAC GPU node, Gemma 3 checkpoint, and `transformers>=4.50.0`.

### Infrastructure smoke (pending)

```bash
sbatch jobs/run_gemma_smoke_test.slurm
```

Validates: CUDA, model load, processor load, chat template, single-image inference, decode.

### Framework smoke (pending)

```bash
sbatch jobs/run_gemma_framework_smoke.slurm
```

Validates: registry → adapter → verifier → parser → `sample_*.json` → `results_index.csv` on 5 A1 samples.

---

## Parity check (design vs Qwen / LLaVA)

| Stage | Qwen2.5-VL | LLaVA-OneVision | Gemma 3 | Match? |
|-------|------------|-----------------|---------|--------|
| Prompt loading | `job.prompt_path.read_text()` | Same | Same | ✓ |
| Image loading | Path in messages / PIL | PIL RGB | Path in messages | ✓ (equivalent) |
| Chat template | `apply_chat_template` | `apply_chat_template` | `apply_chat_template` (tokenize=True) | ✓ |
| Processor call | Separate `processor()` after template | `processor(images, text)` | Combined in `apply_chat_template` | ✓ (HF-native) |
| Generate | `do_sample=False`, 512 tokens | Same | Same | ✓ |
| Decode | Trim input tokens | Trim input tokens | Trim input tokens | ✓ |
| Parser | `parse_verification_response` | Same | Same | ✓ |
| Output JSON | `build_result_record` | Same | Same | ✓ |
| Resume | `VerificationRunner` + `OutputManager` | Same | Same | ✓ |
| System prompt | None | None | None | ✓ |

Gemma follows the **LLaVA adapter pattern** (single-sample `generate_response`, no batch verifier path). Inference API differences are isolated inside `GemmaVerifier`.

---

## Inference implementation notes

Official Hugging Face flow (implemented in `gemma_verifier.py`):

```python
Gemma3ForConditionalGeneration.from_pretrained(..., torch_dtype=bfloat16, device_map="auto")
AutoProcessor.from_pretrained(...)

messages = [{"role": "user", "content": [
    {"type": "image", "image": str(image_path)},
    {"type": "text", "text": prompt},
]}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device, dtype=bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
generated_ids = output_ids[0, input_len:]
response = processor.decode(generated_ids, skip_special_tokens=True)
```

---

## Remaining TODO

| Item | Priority |
|------|----------|
| Download `google/gemma-3-12b-it` to `/deac/csc/yangGrp/luoz23/models/gemma-3-12b-it` | **Required before smoke** |
| Upgrade cluster `transformers` to **≥4.50.0** | **Required before smoke** |
| Run `jobs/run_gemma_smoke_test.slurm` | High |
| Run `jobs/run_gemma_framework_smoke.slurm` | High |
| Confirm JSON parse rate on 5 A1 samples | High |
| Optional: enable `do_pan_and_scan=True` if high-res artifacts appear | Low |
| Full A1–A5 benchmark (`MODEL=gemma`, 1000 samples) | **Out of scope** (not requested) |
| Update `docs/SUPPORTED_MODELS.md` status table | Low |

---

## Explicit non-goals (preserved)

- No evaluation pipeline changes
- No parser changes
- No prompt template changes
- No visualization changes
- No pilot / 1000-sample benchmark runs
- No changes to Qwen or LLaVA behavior

---

See also: [`docs/GEMMA_INTEGRATION_PLAN.md`](GEMMA_INTEGRATION_PLAN.md)
