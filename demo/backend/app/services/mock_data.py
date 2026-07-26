"""In-memory mock experiment catalog (no filesystem reads)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, List, Optional

from shared.models import (
    AblationCode,
    AblationStatistics,
    BoundingBox,
    ConfusionCounts,
    DecisionDistribution,
    ExperimentSummary,
    GroundTruthLabel,
    ModelInfo,
    SampleDetail,
    SampleSummary,
    StatisticsResponse,
    VerificationDecision,
)

DEFAULT_MODEL_KEY = "qwen2_5_vl"
DEFAULT_EXPERIMENT_ID = "20250726_1200"
DEFAULT_ABLATION = AblationCode.A1

ABLATION_CONDITION_NAMES: Dict[AblationCode, str] = {
    AblationCode.A1: "A1_overlay_only",
    AblationCode.A2: "A2_overlay_confidence",
    AblationCode.A3: "A3_overlay_confidence_geometry",
    AblationCode.A4: "A4_overlay_crop_confidence",
    AblationCode.A5: "A5_crop_only",
}

MOCK_MODELS: List[ModelInfo] = [
    ModelInfo(
        model_key="qwen2_5_vl",
        display_name="Qwen2.5-VL",
        description="Primary production VLM for wild palm verification ablations.",
        experiments=[
            ExperimentSummary(
                experiment_id="20250726_1200",
                sample_count=1000,
                ablations=list(AblationCode),
                primary_ablation=AblationCode.A1,
                created_at="2025-07-26T12:00:00Z",
            ),
            ExperimentSummary(
                experiment_id="20250601_0800",
                sample_count=300,
                ablations=[AblationCode.A1, AblationCode.A3, AblationCode.A5],
                primary_ablation=AblationCode.A3,
                created_at="2025-06-01T08:00:00Z",
            ),
        ],
    ),
    ModelInfo(
        model_key="llava",
        display_name="LLaVA-OneVision",
        description="Secondary VLM adapter for comparative ablation studies.",
        experiments=[
            ExperimentSummary(
                experiment_id="20250726_1430",
                sample_count=1000,
                ablations=[AblationCode.A1],
                primary_ablation=AblationCode.A1,
                created_at="2025-07-26T14:30:00Z",
            ),
        ],
    ),
]


@dataclass(frozen=True)
class MockSampleRecord:
    summary: SampleSummary
    yolo_bbox: BoundingBox
    gt_bbox: Optional[BoundingBox]
    confidence_reasoning: str
    visual_reasoning: str


def _sample(
    sample_id: str,
    image_name: str,
    *,
    decision: VerificationDecision,
    matched_gt: bool,
    max_iou: float,
    yolo_confidence: float,
    gt_label: GroundTruthLabel,
    yolo_bbox: BoundingBox,
    gt_bbox: Optional[BoundingBox],
    confidence_reasoning: str,
    visual_reasoning: str,
) -> MockSampleRecord:
    summary = SampleSummary(
        sample_id=sample_id,
        image_name=image_name,
        ablation=DEFAULT_ABLATION,
        decision=decision,
        matched_gt=matched_gt,
        max_iou=max_iou,
        yolo_confidence=yolo_confidence,
        gt_label=gt_label,
    )
    return MockSampleRecord(
        summary=summary,
        yolo_bbox=yolo_bbox,
        gt_bbox=gt_bbox,
        confidence_reasoning=confidence_reasoning,
        visual_reasoning=visual_reasoning,
    )


MOCK_SAMPLES: List[MockSampleRecord] = [
    _sample(
        "sample_000042",
        "100_0003_0001_1.png",
        decision=VerificationDecision.RELIABLE,
        matched_gt=True,
        max_iou=0.82,
        yolo_confidence=0.91,
        gt_label=GroundTruthLabel.POSITIVE,
        yolo_bbox=BoundingBox(x=120.0, y=88.0, width=64.0, height=72.0),
        gt_bbox=BoundingBox(x=118.0, y=90.0, width=66.0, height=70.0),
        confidence_reasoning="High YOLO score; used only as auxiliary context.",
        visual_reasoning="Radial crown with symmetric fronds visible inside the green box.",
    ),
    _sample(
        "sample_000117",
        "100_0012_0004_2.png",
        decision=VerificationDecision.UNRELIABLE,
        matched_gt=False,
        max_iou=0.12,
        yolo_confidence=0.78,
        gt_label=GroundTruthLabel.NEGATIVE,
        yolo_bbox=BoundingBox(x=44.0, y=210.0, width=58.0, height=61.0),
        gt_bbox=None,
        confidence_reasoning="Confidence is high but morphology does not match a palm crown.",
        visual_reasoning="Dense broadleaf canopy without a central crown or frond pattern.",
    ),
    _sample(
        "sample_000203",
        "100_0020_0002_3.png",
        decision=VerificationDecision.UNCERTAIN,
        matched_gt=True,
        max_iou=0.54,
        yolo_confidence=0.63,
        gt_label=GroundTruthLabel.UNCERTAIN,
        yolo_bbox=BoundingBox(x=300.0, y=140.0, width=52.0, height=48.0),
        gt_bbox=BoundingBox(x=298.0, y=142.0, width=55.0, height=50.0),
        confidence_reasoning="Moderate detector confidence with partial occlusion.",
        visual_reasoning="Possible palm crown but overlapping canopy obscures frond structure.",
    ),
    _sample(
        "sample_000318",
        "100_0031_0001_1.png",
        decision=VerificationDecision.RELIABLE,
        matched_gt=True,
        max_iou=0.76,
        yolo_confidence=0.88,
        gt_label=GroundTruthLabel.POSITIVE,
        yolo_bbox=BoundingBox(x=180.0, y=96.0, width=70.0, height=68.0),
        gt_bbox=BoundingBox(x=178.0, y=94.0, width=72.0, height=70.0),
        confidence_reasoning="Strong detector prior; decision driven by crown morphology.",
        visual_reasoning="Clear feather-like fronds radiating from a central point.",
    ),
    _sample(
        "sample_000451",
        "100_0044_0003_2.png",
        decision=VerificationDecision.UNRELIABLE,
        matched_gt=False,
        max_iou=0.08,
        yolo_confidence=0.81,
        gt_label=GroundTruthLabel.NEGATIVE,
        yolo_bbox=BoundingBox(x=92.0, y=320.0, width=60.0, height=55.0),
        gt_bbox=None,
        confidence_reasoning="False positive despite high YOLO confidence.",
        visual_reasoning="Shadowed vegetation patch with no palm-specific texture.",
    ),
]

MOCK_STATISTICS_BY_ABLATION: Dict[AblationCode, AblationStatistics] = {
    AblationCode.A1: AblationStatistics(
        ablation=AblationCode.A1,
        ablation_condition=ABLATION_CONDITION_NAMES[AblationCode.A1],
        dataset_size=1000,
        evaluated_samples=1000,
        precision=0.742,
        recall=0.681,
        f1=0.710,
        accuracy=0.864,
        average_iou=0.512,
        average_confidence=0.734,
        matched_gt_count=612,
        unmatched_gt_count=388,
        decision_distribution=DecisionDistribution(
            reliable_pct=58.4,
            uncertain_pct=11.2,
            unreliable_pct=30.4,
        ),
        confusion_counts=ConfusionCounts(
            true_positive=417,
            false_positive=167,
            true_negative=447,
            false_negative=195,
            uncertain=112,
        ),
    ),
    AblationCode.A3: AblationStatistics(
        ablation=AblationCode.A3,
        ablation_condition=ABLATION_CONDITION_NAMES[AblationCode.A3],
        dataset_size=1000,
        evaluated_samples=998,
        precision=0.768,
        recall=0.694,
        f1=0.729,
        accuracy=0.871,
        average_iou=0.512,
        average_confidence=0.734,
        matched_gt_count=612,
        unmatched_gt_count=388,
        decision_distribution=DecisionDistribution(
            reliable_pct=61.1,
            uncertain_pct=9.8,
            unreliable_pct=29.1,
        ),
        confusion_counts=ConfusionCounts(
            true_positive=428,
            false_positive=152,
            true_negative=442,
            false_negative=184,
            uncertain=98,
        ),
    ),
}


def list_models() -> List[ModelInfo]:
    return MOCK_MODELS


def resolve_model(model_key: str) -> Optional[ModelInfo]:
    return next((model for model in MOCK_MODELS if model.model_key == model_key), None)


def resolve_experiment(model: ModelInfo, experiment_id: str) -> Optional[ExperimentSummary]:
    return next(
        (experiment for experiment in model.experiments if experiment.experiment_id == experiment_id),
        None,
    )


def get_statistics(
    model_key: str,
    experiment_id: str,
    ablation: AblationCode,
) -> Optional[StatisticsResponse]:
    model = resolve_model(model_key)
    if model is None:
        return None
    experiment = resolve_experiment(model, experiment_id)
    if experiment is None:
        return None
    if ablation not in experiment.ablations:
        return None

    statistics = MOCK_STATISTICS_BY_ABLATION.get(ablation)
    if statistics is None:
        # Fall back to A1-shaped mock metrics for ablations without bespoke fixtures.
        base = MOCK_STATISTICS_BY_ABLATION[AblationCode.A1]
        statistics = base.model_copy(
            update={
                "ablation": ablation,
                "ablation_condition": ABLATION_CONDITION_NAMES[ablation],
            }
        )

    return StatisticsResponse(
        model_key=model_key,
        experiment_id=experiment_id,
        statistics=statistics,
    )


def list_sample_summaries(
    *,
    decision: Optional[VerificationDecision] = None,
    matched_gt: Optional[bool] = None,
    gt_label: Optional[GroundTruthLabel] = None,
) -> List[SampleSummary]:
    records = MOCK_SAMPLES
    if decision is not None:
        records = [record for record in records if record.summary.decision == decision]
    if matched_gt is not None:
        records = [record for record in records if record.summary.matched_gt == matched_gt]
    if gt_label is not None:
        records = [record for record in records if record.summary.gt_label == gt_label]
    return [record.summary for record in records]


def get_sample_record(sample_id: str) -> Optional[MockSampleRecord]:
    return next((record for record in MOCK_SAMPLES if record.summary.sample_id == sample_id), None)


def sample_detail(
    record: MockSampleRecord,
    *,
    model_key: str,
    experiment_id: str,
) -> SampleDetail:
    sample_id = record.summary.sample_id
    return SampleDetail(
        **record.summary.model_dump(),
        yolo_bbox=record.yolo_bbox,
        gt_bbox=record.gt_bbox,
        confidence_reasoning=record.confidence_reasoning,
        visual_reasoning=record.visual_reasoning,
        image_path=f"/api/v1/image/{sample_id}",
    )


# 1×1 PNG (green) — placeholder bytes until real overlay files are served.
MOCK_OVERLAY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
