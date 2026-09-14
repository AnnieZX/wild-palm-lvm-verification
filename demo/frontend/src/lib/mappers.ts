/**
 * Map API / shared types into frontend view models.
 */

import type {
  ModelPredictionResult,
  SampleComparisonData,
} from "@/components/comparison/types";
import type {
  ConfusionMatrixData,
  DecisionDistributionCounts,
  ModelComparisonRow,
  PrimaryMetrics,
  PromptComparisonRow,
  StatisticsDashboardData,
} from "@/components/statistics/types";
import type { ViewerBoundingBox } from "@/components/viewer/types";
import { ABLATION_CONDITION_BY_CODE, PROMPT_OPTIONS } from "@/lib/constants";
import type {
  AblationCode,
  AblationStatistics,
  ConfusionCounts,
  ModelInfo,
  SampleDetail,
  SampleDetailResponse,
  StatisticsResponse,
  VerificationDecision,
} from "@shared/types";

export function promptConditionForAblation(ablation: AblationCode): string {
  return (
    ABLATION_CONDITION_BY_CODE[ablation] ??
    PROMPT_OPTIONS.find((option) => option.code === ablation)?.condition ??
    ablation
  );
}

export function sampleToViewerBoxes(sample: SampleDetail): ViewerBoundingBox[] {
  const boxes: ViewerBoundingBox[] = [];

  if (sample.yolo_bbox) {
    boxes.push({
      id: `${sample.sample_id}-yolo`,
      x: sample.yolo_bbox.x,
      y: sample.yolo_bbox.y,
      width: sample.yolo_bbox.width,
      height: sample.yolo_bbox.height,
      role: "yolo",
      label: "YOLO",
      meta: {
        sampleId: sample.sample_id,
        confidence: sample.yolo_confidence ?? "",
        decision: sample.decision ?? "",
      },
    });
  }

  if (sample.gt_bbox) {
    boxes.push({
      id: `${sample.sample_id}-gt`,
      x: sample.gt_bbox.x,
      y: sample.gt_bbox.y,
      width: sample.gt_bbox.width,
      height: sample.gt_bbox.height,
      role: "ground-truth",
      label: "GT",
      meta: {
        sampleId: sample.sample_id,
        matched: sample.matched_gt,
        maxIou: sample.max_iou,
      },
    });
  }

  return boxes;
}

function combineReasoning(sample: SampleDetail): string {
  const parts = [sample.visual_reasoning, sample.confidence_reasoning]
    .map((part) => part.trim())
    .filter(Boolean);
  return parts.join("\n\n");
}

export function sampleResponseToComparisonEntry(
  response: SampleDetailResponse,
  model: ModelInfo,
  promptVersion: string,
): ModelPredictionResult {
  const sample = response.sample;
  const decision: VerificationDecision = sample.decision ?? "Uncertain";

  return {
    modelKey: model.model_key,
    modelName: model.display_name,
    decision,
    confidence: sample.yolo_confidence,
    reasoning: combineReasoning(sample) || "—",
    visualReasoning: sample.visual_reasoning || undefined,
    confidenceReasoning: sample.confidence_reasoning || undefined,
    processingTimeMs: 0,
    promptVersion,
    groundTruth: {
      matched: sample.matched_gt,
      label: sample.gt_label,
      maxIou: sample.max_iou,
    },
  };
}

export function buildSampleComparisonData(
  sample: SampleDetail,
  predictions: ModelPredictionResult[],
): SampleComparisonData {
  return {
    sampleId: sample.sample_id,
    imageName: sample.image_name,
    predictions,
  };
}

function safeRate(numerator: number, denominator: number): number | null {
  if (denominator <= 0) {
    return null;
  }
  return numerator / denominator;
}

/**
 * Derive protocol metrics from confusion counts + decision distribution.
 * Specificity / balanced accuracy / coverage / uncertain rate are not always
 * stored in metrics JSON; when confusion counts exist they are well-defined.
 */
export function deriveProtocolMetrics(
  statistics: AblationStatistics,
): PrimaryMetrics {
  const counts = statistics.confusion_counts;
  const precision = statistics.precision;
  const recall = statistics.recall;
  const f1 = statistics.f1;
  const accuracy = statistics.accuracy;

  const specificity = safeRate(
    counts.true_negative,
    counts.true_negative + counts.false_positive,
  );
  const balancedAccuracy =
    recall !== null && specificity !== null ? (recall + specificity) / 2 : null;

  const definitive =
    counts.true_positive +
    counts.false_positive +
    counts.true_negative +
    counts.false_negative;
  const rateBase =
    statistics.evaluated_samples > 0
      ? statistics.evaluated_samples
      : statistics.dataset_size;
  const coverage = safeRate(definitive, rateBase);
  const uncertainRate = safeRate(counts.uncertain, rateBase);

  return {
    precision,
    recall,
    specificity,
    f1,
    accuracy,
    balanced_accuracy: balancedAccuracy,
    uncertain_rate: uncertainRate,
    coverage,
  };
}

function decisionCountsFromStats(
  statistics: AblationStatistics,
): DecisionDistributionCounts {
  const counts = statistics.confusion_counts;
  // Prefer confusion-derived counts: stored reliable_pct can be relative to a
  // larger CSV than the evaluated subset used for binary metrics.
  const reliable = counts.true_positive + counts.false_positive;
  const unreliable = counts.true_negative + counts.false_negative;
  const uncertain = counts.uncertain;
  if (reliable + unreliable + uncertain > 0) {
    return {
      Reliable: reliable,
      Uncertain: uncertain,
      Unreliable: unreliable,
    };
  }

  const n = statistics.dataset_size;
  const dist = statistics.decision_distribution;
  return {
    Reliable: Math.round((dist.reliable_pct / 100) * n),
    Uncertain: Math.round((dist.uncertain_pct / 100) * n),
    Unreliable: Math.round((dist.unreliable_pct / 100) * n),
  };
}

function confusionMatrixFromCounts(counts: ConfusionCounts): ConfusionMatrixData {
  return {
    row_labels: ["GT+", "GT−"],
    col_labels: ["Pred+", "Pred−"],
    cells: [
      [counts.true_positive, counts.false_negative],
      [counts.false_positive, counts.true_negative],
    ],
    uncertain_excluded: counts.uncertain,
  };
}

function modelRow(
  response: StatisticsResponse,
  models: ModelInfo[],
): ModelComparisonRow {
  const model = models.find((item) => item.model_key === response.model_key);
  const metrics = deriveProtocolMetrics(response.statistics);
  return {
    model_key: response.model_key,
    display_name: model?.display_name ?? response.model_key,
    precision: metrics.precision ?? 0,
    recall: metrics.recall ?? 0,
    f1: metrics.f1 ?? 0,
    accuracy: metrics.accuracy ?? 0,
    specificity: metrics.specificity,
    balanced_accuracy: metrics.balanced_accuracy,
  };
}

function promptRow(response: StatisticsResponse): PromptComparisonRow {
  const metrics = deriveProtocolMetrics(response.statistics);
  return {
    ablation: response.statistics.ablation,
    condition: response.statistics.ablation_condition,
    precision: metrics.precision ?? 0,
    recall: metrics.recall ?? 0,
    f1: metrics.f1 ?? 0,
    accuracy: metrics.accuracy ?? 0,
    specificity: metrics.specificity,
    balanced_accuracy: metrics.balanced_accuracy,
  };
}

export function statisticsToDashboardData(
  primary: StatisticsResponse,
  modelStatistics: StatisticsResponse[],
  ablationStatistics: StatisticsResponse[],
  models: ModelInfo[],
): StatisticsDashboardData {
  const stats = primary.statistics;
  return {
    experiment_id: primary.experiment_id,
    model_key: primary.model_key,
    dataset_size: stats.dataset_size,
    evaluated_samples: stats.evaluated_samples,
    primary_metrics: deriveProtocolMetrics(stats),
    decision_distribution: decisionCountsFromStats(stats),
    confusion_matrix: confusionMatrixFromCounts(stats.confusion_counts),
    model_comparison: modelStatistics.map((entry) => modelRow(entry, models)),
    prompt_comparison: ablationStatistics.map(promptRow),
  };
}
