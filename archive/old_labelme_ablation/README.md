# Old LabelMe Ablation Pipeline (superseded)

These files implemented the original E1–E5 × P1–P6 ablation over **LabelMe GT palms**.
That pipeline was incorrect for the thesis verification path, which now uses **YOLO detections**
from `outputs/verification_dataset/`.

Replaced by:
- `scripts/build_ablation_verification_prompts.py`
- `src/prompts/ablation_verification_prompts.py` (A1–A4 conditions)

## Archived contents

| Path | Role |
|------|------|
| `scripts/prepare_ablation_inputs_100.py` | LabelMe overlay prep |
| `scripts/run_qwen_ablation_100.py` | Old ablation inference |
| `scripts/analyze_ablation_100.py` | Old ablation analysis |
| `scripts/run_qwen_ablation_smoke_test.py` | Old smoke test |
| `jobs/qwen_ablation_100.slurm` | Old SLURM job |
| `jobs/qwen_ablation_smoke_test.slurm` | Old smoke SLURM job |
| `src/prompts/ablation_prompts.py` | P1–P6 prompts |
| `src/lvm/ablation_response_parser.py` | Old response parser |
| `src/preprocessing/ablation_overlay.py` | E1–E5 overlays |

Do not use for new work.
