# 🌴 LVM-Assisted Verification for Wild Palm Detection
### *Reducing manual review burden in large-scale orthomosaic imagery using Large Vision Models*

> **CS Honors Thesis** · Annie Luo · Mentor: Fan Yang · Wake Forest University · 2026

---

## 📌 Overview

Monitoring wild palm populations across large geographic regions is a critical but labor-intensive task. This project builds a **Large Vision Model (LVM)-assisted verification framework** that sits between a YOLO detector and a human reviewer — flagging uncertain predictions and providing interpretable natural-language explanations so experts can focus their attention where it matters most.

Rather than replacing human reviewers, this system **supports them**:

```
Orthomosaic Image → YOLO Detection → LVM Verification → Ranked Review Queue → Human Expert
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                      │
│  ┌────────────────────┐        ┌─────────────────────────────────────┐  │
│  │  21 Full           │        │  JSON Annotations                   │  │
│  │  Orthomosaics      │───────▶│  (Bounding boxes + Seg. masks)      │  │
│  │  (multi-GB each)   │        │  YOLO confidence scores             │  │
│  └────────────────────┘        └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                              │
│                                                                         │
│   Tile-based parsing → Extract image patches → Render overlay           │
│   (original imagery + bounding box/mask overlay per prediction)         │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LVM VERIFICATION MODULE                              │
│                   (Qwen2.5-VL-7B-Instruct)                              │
│                                                                         │
│   Zero-shot / Few-shot prompting                                        │
│   ↓                                                                     │
│   Output per patch:                                                     │
│     • Reliability label:  ✅ reliable │ ⚠️ uncertain │ ❌ unreliable     │
│     • Plain-language explanation (shape, overlap, background context)   │
│     • Verification score (0–1)                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                                     │
│                                                                         │
│   1. Color-coded bounding box overlays (green / amber / red)            │
│   2. Ranked prediction list ordered by verification score               │
│   3. Natural-language explanation for each flagged instance             │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                            👤 Human Expert Review
```

---

## 📊 Dataset

| Subset | Images | Palm Instances | Additional Annotations |
|--------|--------|----------------|------------------------|
| Core dataset | 1,500 | 1,952 | 8,842 bounding boxes (multi-class) |
| New subset | 880 | 5,850 | 5,430 center points · 21,718 endpoints |

- **Ground truth format:** LabelMe JSON (human-labeled)
- **Model predictions:** YOLO-generated candidate detections with confidence scores
- **Source imagery:** 21 full orthomosaics (gigabyte-scale aerial/UAV)

---

## 🔍 Methodology

### 1 · Data Preparation & Preprocessing
Build a tile-based pipeline that parses JSON annotations, extracts image patches centered on each predicted label, and renders overlays combining the original imagery with its corresponding bounding box or segmentation mask.

### 2 · LVM Verification Module
Each patch is submitted to **Qwen2.5-VL-7B-Instruct** via API. The model evaluates:
- Does this region correspond to a valid palm instance?
- What visual cues support or undermine this classification?

Prompting strategies explored: **zero-shot** and **few-shot** (3–5 labeled palm examples).

### 3 · Output Design
Three reviewer-facing outputs:
- **Color-coded overlays** — green (reliable), amber (uncertain), red (unreliable)
- **Ranked list** — predictions sorted by verification score, lowest first
- **Explanations** — natural-language reasoning per flagged instance

### 4 · Evaluation

| Data availability | Approach |
|-------------------|----------|
| Annotated subsets | Quantitative: IoU vs. ground truth · LVM score vs. YOLO confidence |
| Full dataset (no GT) | Qualitative: expert usefulness ratings · consistency testing across repeated queries |

---

## 🤖 Model Choice

**Primary:** [Qwen2.5-VL-7B-Instruct](https://github.com/QwenLM/Qwen2.5-VL) (Apache 2.0)

Chosen for:
- Native bounding-box grounding with structured JSON output
- Strong performance on aerial/UAV imagery tasks
- Runs on single GPU via vLLM; AWQ-quantized variants available
- Active fine-tuning ecosystem (proven for remote-sensing wildfire detection)
- Available in 3B / 7B / 72B — 7B hits the sweet spot for thesis-scale compute

**Backup options considered:**
- InternVL2 / RSCoVLM (remote-sensing specialized)
- GLM-4.1V-9B-Thinking (chain-of-thought reasoning for richer explanations)

---

## 🗓️ Timeline

```
MAY 2026
  Weeks 1–2 ████  Acquire dataset · Dev environment · API access · Preprocessing pipeline

  Weeks 3–5 ████  Integrate Qwen2.5-VL API · Experiment with zero-shot & few-shot prompting

JUNE–JULY 2026
  Weeks 6–8 ████  Overlay rendering · Patch extraction · Prompt engineering refinement

JULY–AUGUST 2026
  Weeks 9–10 ███  Stability & consistency testing (repeated queries, same inputs)

AUGUST 2026
  Weeks 11–12 ██  Module fusion · Confidence scoring system · Reviewer output interface

FALL 2026
  Sep  (Wks 1–4)  Debug & refine based on early expert feedback
  Oct  (Wks 5–8)  Human-centered evaluation · Structured reviewer feedback collection
  Nov–Dec (Wks 9–12)  Write thesis report · Prepare final presentation
```

**Expected workload:** ~10–15 hrs/week (summer) · ~6–8 hrs/week (fall semester)

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Object detection | YOLOv8 (pre-trained predictions) |
| LVM inference | Qwen2.5-VL-7B-Instruct |
| Inference serving | vLLM |
| Image processing | Python · OpenCV · Pillow |
| Annotation format | LabelMe JSON |
| Evaluation | IoU (spatial) · Human ratings (qualitative) |
| IDE | VS Code + Jupyter extension |

---

## 📁 Project Structure

```
palm-lvm-verification/
├── data/
│   ├── raw/                  # Original orthomosaics (symlinked, not committed)
│   ├── annotations/          # LabelMe JSON ground truth
│   └── yolo_predictions/     # YOLO candidate detections + confidence scores
├── src/
│   ├── preprocessing/
│   │   ├── tile_extractor.py     # Tile-based patch extraction pipeline
│   │   └── overlay_renderer.py   # Bounding box / mask overlay rendering
│   ├── verification/
│   │   ├── lvm_client.py         # Qwen2.5-VL API wrapper
│   │   ├── prompt_strategies.py  # Zero-shot & few-shot prompt templates
│   │   └── scorer.py             # Reliability label + score aggregation
│   └── output/
│       ├── reviewer_interface.py # Ranked list + color-coded export
│       └── visualizer.py         # Overlay visualization tools
├── evaluation/
│   ├── iou_eval.py           # Quantitative IoU evaluation
│   ├── consistency_test.py   # Repeated-query stability testing
│   └── expert_feedback/      # Structured reviewer rating collection
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_prompt_experiments.ipynb
│   └── 03_evaluation_results.ipynb
├── tests/
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/annieluo/palm-lvm-verification.git
cd palm-lvm-verification

# 2. Create environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure model access (Qwen2.5-VL via vLLM or API)
cp .env.example .env
# Edit .env with your API endpoint / local model path

# 5. Run preprocessing on a sample
python src/preprocessing/tile_extractor.py \
  --annotation data/annotations/sample.json \
  --output data/patches/

# 6. Run LVM verification on extracted patches
python src/verification/lvm_client.py \
  --patches data/patches/ \
  --strategy few_shot \
  --output results/verification_output.json

# 7. Generate reviewer output
python src/output/reviewer_interface.py \
  --input results/verification_output.json \
  --output results/review_queue/
```

---

## 📖 References

1. Kuckreja et al. (2024). *GeoChat: Grounded Large Vision-Language Model for Remote Sensing.* CVPR 2024.
2. Hu et al. (2023). *Vision-Language Models in Remote Sensing: Current Progress and Future Trends.* arXiv:2305.05726.
3. Syetiawan et al. (2025). *Deep Learning-Based Palm Tree Detection in UAV Imagery with Mask RCNN.* TELKOMNIKA, 23(1).
4. Mazzia et al. (2021). *Deep-Learning-Based Automated Palm Tree Counting and Geolocation.* Agronomy, 11(8).
5. Bai et al. (2025). *Qwen2.5-VL Technical Report.* arXiv:2502.13923.

---

## 👤 Author

**Annie Luo** · CS Honors Thesis  
Mentor: **Fan Yang**  
Wake Forest University · May 2026

---

*This project is part of ongoing research into scalable, human-in-the-loop ecological monitoring using large vision models.*
