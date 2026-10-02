# Model-expansion inference campaign (A1–A5 @5747)

Expands the frozen A1–A5 behavior study along within-family parameter scaling and
cross-family diversity. Starting checkpoint: `6f95e870a74ffcfec0072a6362e9f5eb051f6ec2`.

Frozen and unchanged: A1–A5 semantics, prompt text, shared parser, evaluator,
Protocol-v2 outputs, existing predictions, semantic-review labels, completed runs.
Completed anchors are **not** rerun.

## Integration pattern

- One canonical key per parameter size (e.g. `qwen3_vl_4b`), its own
  `configs/models/<key>.yaml` (pinned `revision`, `dtype`, `attn_implementation`,
  generation settings) and its own output namespace `outputs/verification/<key>/`.
- Sizes reuse the family adapter and environment. Adapters read size settings via
  `src/lvm/model_settings.py`; `jobs/lib/model_runtime.sh::model_family` maps
  size keys to the family env.
- Checkpoints: `scripts/pipeline/download_pinned_model.py` (pinned SHA, refuses to
  overwrite a different repo/revision, verifies every upstream file and size).
- Each sample's `generation` block records requested and resolved attention backend,
  checkpoint revision and git commit. Each job writes
  `outputs/verification/<key>/<exp>/_run_metadata/<cond>_<jobid>.json` and a GPU-memory log.

## Technical qualification: fixed GT-independent paired probe

- Inputs: `outputs/verification_ablation_100/<A1..A5>/`, sample_000001–sample_000100
  (first 100 detections by index order; selection does not use GT). Same 100 IDs in
  all five conditions; prompt and image files are byte-identical to the @5747 inputs.
- A1's first 10 samples are re-run in a separate process (determinism check).
- Job: `jobs/run_paired_probe_qual.slurm`; verdict:
  `scripts/evaluate_paired_probe_qualification.py` →
  `outputs/diagnostics/model_qualification/paired_probe/<key>/<exp>/`.
- Hard gates: all probe records present, 0 inference errors, ≥95% usable parsed output
  in every condition, all parsed labels valid, 0 garbage outputs, revision matches pin,
  one resolved attention backend.
- Reported only (never gated): decision distribution, label dominance, Uncertain usage,
  A1→Ak transitions, cross-condition responsiveness, determinism agreement, runtime,
  peak GPU memory. No GT, alignment, or semantic metric is used.
- Full submission: `scripts/submit_qualified_model.sh` (requires PASS verdict and a
  clean, pushed tree) → `scripts/submit_model_ablation.sh` → `jobs/run_verification.slurm`;
  five independent jobs, N=5747 each. Submissions are logged in
  `logs/campaign/submissions.csv`.

## Attention backend

New scaling runs use SDPA (`attn_implementation: sdpa`). FlashAttention is not installed.
The Qwen3-VL-8B and InternVL3.5-8B-HF anchors ran with `attn_implementation: null` under
transformers 4.57.6 without flash-attn, which resolves to SDPA, so each ladder shares one
backend.

## Environments

- Qwen3-VL sizes: `/deac/csc/yangGrp/luoz23/envs/wild-palm-qwen3vl` (transformers 4.57.6, torch 2.8.0+cu128), same as the 8B anchor.
- InternVL3.5-HF sizes: `/usr/bin/python` 3.9 (transformers 4.57.6), same interpreter as the 8B anchor;
  `model_runtime.sh` pins it for the family and asserts the transformers version.
- Ministral 3 sizes: `/deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4` (transformers 5.17.0,
  torch 2.11.0+cu128), the env the 8B integration already used; reused unchanged.

## Ministral 3 (2512, BF16 checkpoints)

- Unquantized `-2512-BF16` repos for all sizes (the non-BF16 repos are FP8-quantized).
  Duplicate Mistral-native `consolidated.safetensors`, `params.json`, `tekken.json` are not
  downloaded (HF-format shards are complete).
- The 8B interface failure (literal newlines inside `visual_reasoning`) is handled by the
  permitted model-specific raw-response normalizer documented in
  [`FRAMEWORK_FREEZE.md`](FRAMEWORK_FREEZE.md#amendment-2026-10-01--ministral-3-raw-response-normalizer).
  On the historical Stage 0 responses it recovers 7/7 failures, identical to
  `json.loads(strict=False)`. All seven failures were Unreliable responses, so without it
  parse failures would be label-biased.
- 8B is requalified on the paired probe like every new size.

## Ledger

| Model key | Family / size | Revision | GPU | Integration commit | Qualification | A1 | A2 | A3 | A4 | A5 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3_vl` | Qwen3-VL / 8B | `0c351dd0` | L40S | (historical) | anchor | 8349703 | 8349704 | 8349705 | 8349706 | 8349707 | Complete — do not rerun |
| `qwen3_vl_4b` | Qwen3-VL / 4B | `ebb281ec` | L40S | pending | pending | — | — | — | — | — | integrating |
| `qwen3_vl_2b` | Qwen3-VL / 2B | `89644892` | L40S | pending | pending | — | — | — | — | — | integrating |
| `qwen3_vl_32b` | Qwen3-VL / 32B | `0cfaf481` | H200 | pending | pending | — | — | — | — | — | integrating |
| `internvl3_5_hf` | InternVL3.5-HF / 8B | `741a7d03` | L40S | (historical) | anchor | 8351993 | 8351994 | 8351995 | 8351996 | 8351997 | Complete — do not rerun |
| `internvl3_5_hf_4b` | InternVL3.5-HF / 4B | `6bd44874` | L40S | pending | pending | — | — | — | — | — | integrating |
| `internvl3_5_hf_14b` | InternVL3.5-HF / 14B | `226b96d5` | L40S | pending | pending | — | — | — | — | — | integrating |
| `internvl3_5_hf_2b` | InternVL3.5-HF / 2B | `3f301ffc` | L40S | pending | pending | — | — | — | — | — | integrating |
| `ministral3_8b` | Ministral 3 / 8B | `f6fae979` | L40S | pending | requalify | — | — | — | — | — | integrating |
| `ministral3_3b` | Ministral 3 / 3B | `b6d637be` | L40S | pending | pending | — | — | — | — | — | integrating |
| `ministral3_14b` | Ministral 3 / 14B | `3cea74c1` | L40S | pending | pending | — | — | — | — | — | integrating |
