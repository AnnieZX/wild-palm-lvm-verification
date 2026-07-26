"""Pydantic models for the Wild Palm Verification Demo API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


class VerificationDecision(str, Enum):
    RELIABLE = "Reliable"
    UNCERTAIN = "Uncertain"
    UNRELIABLE = "Unreliable"


class AblationCode(str, Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"


class BoundingBox(BaseModel):
    x: float = Field(..., description="Top-left x in pixels")
    y: float = Field(..., description="Top-left y in pixels")
    width: float = Field(..., ge=0, description="Box width in pixels")
    height: float = Field(..., ge=0, description="Box height in pixels")


class GroundTruthLabel(str, Enum):
    """Derived GT category used for qualitative review."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["wild-palm-demo-backend"])


class ApiError(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


class ExperimentSummary(BaseModel):
    experiment_id: str = Field(..., examples=["20250726_1200"])
    sample_count: int = Field(..., ge=0)
    ablations: list[AblationCode]
    primary_ablation: AblationCode = Field(
        default=AblationCode.A1,
        description="Default ablation shown in the demo UI",
    )
    created_at: str = Field(..., description="ISO-8601 timestamp")


class ModelInfo(BaseModel):
    model_key: str = Field(..., examples=["qwen2_5_vl"])
    display_name: str = Field(..., examples=["Qwen2.5-VL"])
    description: str = Field(
        default="",
        description="Short model description for the demo catalog",
    )
    experiments: list[ExperimentSummary]


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# GET /statistics
# ---------------------------------------------------------------------------


class DecisionDistribution(BaseModel):
    reliable_pct: float = Field(..., ge=0, le=100)
    uncertain_pct: float = Field(..., ge=0, le=100)
    unreliable_pct: float = Field(..., ge=0, le=100)


class ConfusionCounts(BaseModel):
    true_positive: int = Field(..., ge=0)
    false_positive: int = Field(..., ge=0)
    true_negative: int = Field(..., ge=0)
    false_negative: int = Field(..., ge=0)
    uncertain: int = Field(..., ge=0)


class AblationStatistics(BaseModel):
    ablation: AblationCode
    ablation_condition: str = Field(
        ...,
        examples=["A1_overlay_only"],
        description="Full ablation folder name from experiment outputs",
    )
    dataset_size: int = Field(..., ge=0)
    evaluated_samples: int = Field(..., ge=0)
    precision: float | None = Field(None, ge=0, le=1)
    recall: float | None = Field(None, ge=0, le=1)
    f1: float | None = Field(None, ge=0, le=1)
    accuracy: float | None = Field(None, ge=0, le=1)
    average_iou: float = Field(..., ge=0, le=1)
    average_confidence: float = Field(..., ge=0, le=1)
    matched_gt_count: int = Field(..., ge=0)
    unmatched_gt_count: int = Field(..., ge=0)
    decision_distribution: DecisionDistribution
    confusion_counts: ConfusionCounts


class StatisticsResponse(BaseModel):
    model_key: str
    experiment_id: str
    statistics: AblationStatistics


# ---------------------------------------------------------------------------
# GET /samples, GET /sample/{sample_id}
# ---------------------------------------------------------------------------


class SampleSummary(BaseModel):
    sample_id: str = Field(..., examples=["sample_000042"])
    image_name: str = Field(..., examples=["100_0003_0001_1.png"])
    ablation: AblationCode
    decision: VerificationDecision | None = Field(
        None,
        description="VLM verification decision; null if not yet evaluated",
    )
    matched_gt: bool
    max_iou: float = Field(..., ge=0, le=1)
    yolo_confidence: float | None = Field(None, ge=0, le=1)
    gt_label: GroundTruthLabel


class SampleListResponse(BaseModel):
    model_key: str
    experiment_id: str
    ablation: AblationCode
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1, le=200)
    samples: list[SampleSummary]


class SampleDetail(SampleSummary):
    yolo_bbox: BoundingBox | None = None
    gt_bbox: BoundingBox | None = None
    confidence_reasoning: str = ""
    visual_reasoning: str = ""
    image_path: str = Field(
        ...,
        description="Relative API path to fetch the overlay image for this sample",
        examples=["/api/v1/image/sample_000042"],
    )


class SampleDetailResponse(BaseModel):
    model_key: str
    experiment_id: str
    sample: SampleDetail


# ---------------------------------------------------------------------------
# GET /image/{sample_id}
# ---------------------------------------------------------------------------


class ImageResponseMeta(BaseModel):
    """Documented response headers/body metadata for overlay images."""

    sample_id: str
    content_type: str = Field(default="image/png")
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    ablation: AblationCode
    model_key: str
    experiment_id: str
