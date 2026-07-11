# Architecture

System architecture for the wild palm VLM verification framework (frozen July 2026).

Related documents: [`FRAMEWORK_FREEZE.md`](FRAMEWORK_FREEZE.md) · [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) · [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md)

---

## Diagram 1 — Overall System

End-to-end flow from raw imagery through detection, verification, evaluation, and visualization.

```mermaid
graph TD
    RAW[Raw Orthomosaic Patches]
    LM[LabelMe Annotations]
    YOLO[YOLO Detection<br/>run_full_inference.py]
    PRED[predictions_full.json]

    DS[Verification Dataset Generation]
    VDS[verification_dataset/]
    ABL[A1–A5 Ablation Builder]
    AIN[verification_ablation_N/]

    CLI[run_verification.py]
    RUN[VerificationRunner]
    REG[Registry]
    ADP[Model Adapter]
    VER[Model Verifier]
    OUT[Output Manager]
    RES[sample_*.json + results_index.csv]

    EVAL[Ground Truth Evaluation]
    MET[Metrics]
    VIZ[Visualization]

    RAW --> YOLO
    YOLO --> PRED
    PRED --> DS
    DS --> VDS
    VDS --> ABL
    ABL --> AIN
    AIN --> CLI
    CLI --> RUN
    RUN --> REG
    REG --> ADP
    ADP --> VER
    VER --> OUT
    OUT --> RES
    RES --> EVAL
    LM --> EVAL
    PRED --> EVAL
    EVAL --> MET
    MET --> VIZ
    RES --> VIZ
```

**Key separation:** YOLO produces detections. The verification framework judges them. LabelMe GT enters only at evaluation.

---

## Diagram 2 — Verification Framework

Frozen model-agnostic layer under `src/verification/`.

```mermaid
graph TD
    CLI[scripts/run_verification.py]
    CFG[Model Config<br/>configs/models/*.yaml]
    JOBS[VerificationJob loader<br/>jobs.py]
    RUN[VerificationRunner<br/>runner.py]
    RESUME[Resume filter<br/>verification_resume.py]
    REG[Registry<br/>registry.py]
    ADP[BaseVerificationAdapter]
    OUT[VerificationOutputManager<br/>output_manager.py]
    REC[build_result_record<br/>records.py]
    IDX[results_index.csv]

    CLI --> CFG
    CLI --> JOBS
    CLI --> REG
    REG --> ADP
    CLI --> RUN
    RUN --> RESUME
    RESUME --> RUN
    RUN --> ADP
    ADP --> REC
    REC --> OUT
    OUT --> IDX
```

| Component | Frozen? | Role |
|-----------|---------|------|
| VerificationRunner | Yes | Orchestration, resume, persistence |
| VerificationJob | Yes | `(sample_id, image_path, prompt_path)` |
| BaseVerificationAdapter | Yes | `verify(job) → VerificationOutcome` |
| Registry | Yes | `create_adapter(model, **kwargs)` |
| OutputManager | Yes | JSON + CSV index |
| build_result_record | Yes | Canonical output schema |

---

## Diagram 3 — Inference Pipeline

Per-sample path from job to persisted result.

```mermaid
graph LR
    JOB[VerificationJob]
    ADP[Adapter.verify]
    VER[Verifier<br/>Transformers inference]
    RAW[Raw model text]
    CLN[Cleanup Layer<br/>parsers/cleanup.py]
    PAR[Shared Parser<br/>parsers/base.py]
    REC[build_result_record]
    JSON[sample_*.json]
    CSV[results_index.csv]

    JOB --> ADP
    ADP --> VER
    VER --> RAW
    RAW --> CLN
    CLN --> PAR
    PAR --> REC
    REC --> JSON
    JSON --> CSV
```

**Parser pipeline (frozen):**

```
normalize_raw_response() → parse_json_response() → normalize_decision()
```

Allowed decisions: `Reliable`, `Uncertain`, `Unreliable`.

---

## Diagram 4 — Evaluation Pipeline

Quantitative assessment against LabelMe ground truth.

```mermaid
graph TD
    RES[Verification results<br/>sample_*.json]
    IDX[verification_dataset/index.csv]
    PRED[YOLO predictions_full.json]
    GT[LabelMe JSON<br/>label == palm]

    EVAL[evaluate_verification_against_groundtruth.py]
    MATCH[Greedy IoU matching<br/>gt_matching.py]
    CSV[A1_evaluation.csv]
    MET[compute_verification_metrics.py]
    MJSON[A1_metrics.json]

    RES --> EVAL
    IDX --> EVAL
    PRED --> EVAL
    GT --> EVAL
    EVAL --> MATCH
    MATCH --> CSV
    CSV --> MET
    MET --> MJSON
```

**Matching rules (frozen):**

- Per-image greedy one-to-one assignment
- Sort candidate pairs by descending IoU
- Accept match when IoU ≥ 0.5
- Matched detection → GT positive; unmatched → GT negative
- Uncertain predictions excluded from binary Precision / Recall / F1

---

## Diagram 5 — Multi-Model Architecture

How new VLMs plug into the frozen framework.

```mermaid
graph TD
    CLI[run_verification.py<br/>--model &lt;key&gt;]
    REG[Registry]
    CFG[configs/models/&lt;key&gt;.yaml]

    subgraph implemented [Implemented]
        QW[qwen2_5_vl]
        QV[qwen_verifier.py]
        QA[qwen_verification_adapter.py]
    end

    subgraph planned [Planned]
        LV[llava]
        GM[gemma4]
        Q3[qwen3_vl]
    end

    SHARED[Shared components]
    DS[Same verification dataset]
    PR[Same A1–A5 prompts]
    PA[Shared response parser]
    EV[Same evaluation protocol]

    CLI --> REG
    REG --> CFG
    REG --> QW
    QW --> QV
    QW --> QA
    REG -.-> LV
    REG -.-> GM
    REG -.-> Q3

    QA --> SHARED
    LV -.-> SHARED
    GM -.-> SHARED
    Q3 -.-> SHARED
    SHARED --> DS
    SHARED --> PR
    SHARED --> PA
    SHARED --> EV
```

**Per-model output isolation:**

```
outputs/verification/<registry_key>/<experiment_id>/A1/
outputs/evaluation/<registry_key>/<experiment_id>/A1/
```

Legacy: `outputs/verification/qwen/` (pre-freeze) remains readable.

---

## Pipeline Diagrams

Publication-quality views of each major pipeline stage.

### 1. Dataset Generation Pipeline

```mermaid
flowchart TD
    A[Raw patch PNG + LabelMe JSON]
    B[YOLO predictions_full.json]
    C{confidence ≥ threshold?}
    D[Extract bbox + metadata]
    E[Render overlay image<br/>dimmed background + green bbox]
    F[Write metadata JSON]
    G[Write default prompt]
    H[Append index.csv row]
    I[verification_dataset/]

    A --> D
    B --> C
    C -->|yes| D
    C -->|no| X[Skip detection]
    D --> E
    D --> F
    E --> I
    F --> I
    G --> I
    H --> I
```

**Output:** One verification sample per accepted YOLO detection — overlay image, metadata, prompt, and index entry.

---

### 2. Prompt Generation Pipeline

```mermaid
flowchart TD
    A[verification_dataset/index.csv]
    B[For each sample × condition A1–A5]
    C[Select image variant]
    D[Build metadata block per condition]
    E[build_ablation_verification_prompt]
    F[Write prompt .txt]
    G[Write condition-specific image if needed]
    H[Write prompt_index.csv]
    I[verification_ablation_N/]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    F --> H
    G --> H
    H --> I
```

**Fairness:** Same semantic prompt template across conditions; only image input and metadata sections vary.

---

### 3. Verification Inference Pipeline

```mermaid
flowchart TD
    A[prompt_index.csv]
    B[Load VerificationJobs]
    C{--resume?}
    D[Filter completed sample_ids]
    E[VerificationRunner.run]
    F[Adapter.verify per job]
    G[Load image + prompt .txt]
    H[Model inference GPU]
    I[Cleanup + shared parser]
    J[build_result_record]
    K[save_json + finalize_index]
    L[results_index.csv]

    A --> B
    B --> C
    C -->|yes| D
    C -->|no| E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
```

**Batching:** `RunnerConfig.batch_size` groups progress logging; adapters may batch internally.

---

### 4. Output Pipeline

```mermaid
flowchart TD
    A[VerificationOutcome]
    B[build_result_record]
    C{status}
    D[sample_XXXXXX.json]
    E[results_index.csv row]
    F[Per-condition directory]
    G[Per-experiment directory]
    H[Per-model directory]

    A --> B
    B --> C
    C -->|ok / parse_error / inference_error| D
    D --> E
    E --> F
    F --> G
    G --> H

    subgraph schema [Frozen JSON fields]
        S1[sample_id, decision]
        S2[confidence_reasoning, visual_reasoning]
        S3[raw_response, parsed_response]
        S4[runtime_seconds, parse_error, inference_error]
        S5[timestamp + optional model_key, condition]
    end

    B --> schema
```

**Evaluation reads only `decision`.** All other fields support audit, visualization, and debugging.

---

### 5. Evaluation Pipeline

```mermaid
flowchart TD
    A[sample_*.json]
    B[Read decision field]
    C[Load YOLO bbox from index]
    D[Load LabelMe GT palms]
    E[Compute pairwise IoU]
    F[Sort pairs by IoU desc]
    G[Greedy one-to-one match]
    H{IoU ≥ 0.5?}
    I[Assign matched_gt polarity]
    J[Merge decision + GT]
    K[A1_evaluation.csv]
    L[compute_verification_metrics]
    M{decision == Uncertain?}
    N[Exclude from binary metrics]
    O[Precision, Recall, F1, Accuracy]
    P[A1_metrics.json]

    A --> B
    A --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H -->|yes| I
    H -->|no| I
    I --> J
    B --> J
    J --> K
    K --> L
    L --> M
    M -->|yes| N
    M -->|no| O
    N --> P
    O --> P
```

---

### 6. Cross-Model Framework

```mermaid
flowchart TB
    subgraph fixed [Fixed across all models]
        DS[verification_dataset]
        AB[A1–A5 ablation inputs]
        PR[Prompt semantic content]
        SC[Output JSON schema]
        GT[Greedy IoU 0.5 matching]
        MT[Metrics formulas]
        UC[Uncertain exclusion]
        RS[Resume by sample_id]
    end

    subgraph variable [Variable per model]
        W[Checkpoint weights]
        VE[Vision encoder]
        TK[Processor / tokenizer]
        CT[Chat template wrapping]
        GP[Generation parameters]
        RF[Raw response format]
    end

    subgraph models [Model backends]
        M1[qwen2_5_vl ✓]
        M2[llava]
        M3[gemma4]
        M4[qwen3_vl]
    end

    fixed --> models
    variable --> models
    models --> OUT[Isolated output directories]
    OUT --> CMP[Fair cross-model comparison]
```

---

### 7. Repository Layer Diagram

```mermaid
graph TB
    subgraph entry [Entry Layer]
        SCR[scripts/]
        JOB[jobs/]
    end

    subgraph framework [Frozen Framework Layer]
        VER[src/verification/]
        UTL[src/utils/verification_resume.py]
        CFG[src/config/model_config.py]
        PTH[src/paths.py]
    end

    subgraph models [Model Layer]
        LVM[src/lvm/]
        PAR[src/lvm/parsers/]
    end

    subgraph data [Data Layer]
        PRE[src/preprocessing/]
        PRM[src/prompts/]
        YO[src/yolo/]
    end

    subgraph analysis [Analysis Layer]
        EVL[src/evaluation/]
        EVS[scripts/evaluate_*.py]
        MET[scripts/compute_verification_metrics.py]
    end

    subgraph viz [Visualization Layer]
        VIZ[src/visualization/]
        VSC[scripts/visualization/]
    end

    subgraph config [Configuration]
        CFS[configs/models/]
    end

    subgraph artifacts [Artifacts]
        OUT[outputs/]
    end

    SCR --> VER
    SCR --> LVM
    JOB --> SCR
    VER --> LVM
    LVM --> PAR
    VER --> OUT
    SCR --> PRE
    PRE --> OUT
    PRM --> PRE
    EVS --> EVL
    EVS --> OUT
    MET --> OUT
    VIZ --> OUT
    VSC --> VIZ
    CFG --> CFS
    PTH --> OUT
```

---

## Ablation orchestration (cluster)

```mermaid
sequenceDiagram
    participant User
    participant Submit as submit_qwen_ablation.sh
    participant Slurm as run_qwen_ablation.slurm
    participant Orch as run_qwen_ablation_experiment.sh
    participant Inf as run_ablation_verification.py
    participant Eval as evaluate + metrics

    User->>Submit: SAMPLE_SIZE, MODEL
    Submit->>Slurm: sbatch
    loop A1 through A5
        Slurm->>Orch: run condition
        Orch->>Inf: inference + resume
        Inf->>Eval: per-condition evaluation
    end
    Orch->>User: experiment summary
```

---

*Last updated: July 2026 (architecture freeze)*
