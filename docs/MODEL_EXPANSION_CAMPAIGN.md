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
- Qwen2.5-VL sizes: `/usr/bin/python` 3.9 (transformers 4.57.6, qwen-vl-utils), the interpreter
  the 7B anchor logged; its transformers version at the time (July 2026) was not logged. `dtype: auto`
  as for the anchor (resolves to bfloat16; recorded as `torch_dtype_resolved`). The 7B anchor's
  revision was not recorded; its local shard sizes match upstream `cc594898`.
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

## Partial models (MiniCPM-V-4.5, Molmo2-8B)

- A1@5747 is complete for both (`8351080`, `8351081`, H200; 5747/5747, 0 parse/inference
  errors) and is not rerun. Adapter/verifier code is unchanged since the A1 runs except
  metadata-only run provenance added to the `generation` block (commit `887b666`).
- Integration check: paired probe on L40S with `REFERENCE_DIRS` reporting probe-vs-stored
  agreement on the shared IDs against A1@1000 (L40S) and A1@5747 (H200).
- After PASS, A2–A5 go into the existing experiment directory on H200 (hardware parity with
  A1@5747): `submit_qualified_model.sh <key> <report> --experiment-id <existing>
  --conditions A2,A3,A4,A5 --add-to-existing --partition gpu_small --gres gpu:H200_141:1`.

## Llama-3.2-11B-Vision — held (decision required)

Existing smoke outputs (20 samples × A1–A5 + Stage 0): 13/20 smoke responses ignore the JSON
contract and answer in prose ending with `**Answer:** <label>` / `**Decision:** <label>`; no
`{` appears. An explicit-marker extractor would recover 14/15 failures with exactly one label
(one has only a bare label mention). This is not JSON repair: it synthesizes a decision from
free text with no `confidence_reasoning` / `visual_reasoning`, so it is held under the
"questionable parsing" stop condition pending a decision. Not submitted.

## Ledger

Full-run submissions are also appended to `logs/campaign/submissions.csv` (job id, commit,
revision, results dir).

| Model key | Family / size | Revision | GPU | Integration commit | Qualification | A1 | A2 | A3 | A4 | A5 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3_vl` | Qwen3-VL / 8B | `0c351dd0` | L40S | (historical) | anchor | 8349703 | 8349704 | 8349705 | 8349706 | 8349707 | Complete — do not rerun |
| `qwen3_vl_4b` | Qwen3-VL / 4B | `ebb281ec` | L40S | `7fd59f5` (submitted at `887b666`) | PASS (probe 8365191: 100% usable, sdpa, det 10/10, 14.5 GB, 58/100 responsive) | 8365266 | 8365267 | 8365268 | 8365269 | 8365270 | queued |
| `qwen3_vl_2b` | Qwen3-VL / 2B | `89644892` | L40S | `7fd59f5` | probe 8365192 | — | — | — | — | — | qualifying |
| `qwen3_vl_32b` | Qwen3-VL / 32B | `0cfaf481` | H200 | `7fd59f5` | probe 8365193 (waiting for H200) | — | — | — | — | — | qualifying |
| `internvl3_5_hf` | InternVL3.5-HF / 8B | `741a7d03` | L40S | (historical) | anchor | 8351993 | 8351994 | 8351995 | 8351996 | 8351997 | Complete — do not rerun |
| `internvl3_5_hf_4b` | InternVL3.5-HF / 4B | `6bd44874` | L40S | `fa0b019` | probe 8365195 | — | — | — | — | — | qualifying |
| `internvl3_5_hf_14b` | InternVL3.5-HF / 14B | `226b96d5` | L40S | `fa0b019` | probe 8365196 | — | — | — | — | — | qualifying |
| `internvl3_5_hf_2b` | InternVL3.5-HF / 2B | `3f301ffc` | L40S | `fa0b019` | probe 8365197 | — | — | — | — | — | qualifying |
| `ministral3_8b` | Ministral 3 / 8B | `f6fae979` | L40S | `a53d89c` | probe 8365214 | — | — | — | — | — | qualifying |
| `ministral3_3b` | Ministral 3 / 3B | `b6d637be` | L40S | `a53d89c` | probe 8365215 | — | — | — | — | — | qualifying |
| `ministral3_14b` | Ministral 3 / 14B | `3cea74c1` | L40S | `a53d89c` | probe 8365216 | — | — | — | — | — | qualifying |
| `qwen2_5_vl` | Qwen2.5-VL / 7B | unrecorded | L40S | (historical) | anchor | historical | | | | 8318977 (A5 resume) | Complete — do not rerun |
| `qwen2_5_vl_3b` | Qwen2.5-VL / 3B | `66285546` | L40S | `ff7a75e` | probe 8365240 | — | — | — | — | — | qualifying |
| `qwen2_5_vl_32b` | Qwen2.5-VL / 32B | `7cfb30d7` | H200 | `ff7a75e` | probe 8365241 (waiting for H200) | — | — | — | — | — | qualifying |
| `minicpm_v4_5` | MiniCPM-V / 4.5 (8.7B) | `daef484c` | H200 (A1) | `887b666` | probe 8365263 (+ A1 reference agreement) | 8351080 (complete) | — | — | — | — | A1 done; A2–A5 after probe |
| `molmo2_8b` | Molmo2 / 8B | `e28fa285` | H200 (A1) | `887b666` | probe 8365264 (+ A1 reference agreement) | 8351081 (complete) | — | — | — | — | A1 done; A2–A5 after probe |
| `llama3_2_11b_vision` | Llama 3.2 Vision / 11B | — | — | — | held (free-text decisions) | — | — | — | — | — | awaiting decision |
