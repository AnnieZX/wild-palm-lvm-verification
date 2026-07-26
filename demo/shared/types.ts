/**
 * TypeScript interfaces mirroring demo/shared/models.py.
 */

export type VerificationDecision = "Reliable" | "Uncertain" | "Unreliable";

export type AblationCode = "A1" | "A2" | "A3" | "A4" | "A5";

export type GroundTruthLabel = "positive" | "negative" | "uncertain";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface ApiError {
  detail: string;
}

export interface ExperimentSummary {
  experiment_id: string;
  sample_count: number;
  ablations: AblationCode[];
  primary_ablation: AblationCode;
  created_at: string;
}

export interface ModelInfo {
  model_key: string;
  display_name: string;
  description: string;
  experiments: ExperimentSummary[];
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface DecisionDistribution {
  reliable_pct: number;
  uncertain_pct: number;
  unreliable_pct: number;
}

export interface ConfusionCounts {
  true_positive: number;
  false_positive: number;
  true_negative: number;
  false_negative: number;
  uncertain: number;
}

export interface AblationStatistics {
  ablation: AblationCode;
  ablation_condition: string;
  dataset_size: number;
  evaluated_samples: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  accuracy: number | null;
  average_iou: number;
  average_confidence: number;
  matched_gt_count: number;
  unmatched_gt_count: number;
  decision_distribution: DecisionDistribution;
  confusion_counts: ConfusionCounts;
}

export interface StatisticsResponse {
  model_key: string;
  experiment_id: string;
  statistics: AblationStatistics;
}

export interface SampleSummary {
  sample_id: string;
  image_name: string;
  ablation: AblationCode;
  decision: VerificationDecision | null;
  matched_gt: boolean;
  max_iou: number;
  yolo_confidence: number | null;
  gt_label: GroundTruthLabel;
}

export interface SampleListResponse {
  model_key: string;
  experiment_id: string;
  ablation: AblationCode;
  total: number;
  page: number;
  page_size: number;
  samples: SampleSummary[];
}

export interface SampleDetail extends SampleSummary {
  yolo_bbox: BoundingBox | null;
  gt_bbox: BoundingBox | null;
  confidence_reasoning: string;
  visual_reasoning: string;
  image_path: string;
}

export interface SampleDetailResponse {
  model_key: string;
  experiment_id: string;
  sample: SampleDetail;
}

export interface ImageResponseMeta {
  sample_id: string;
  content_type: string;
  width: number;
  height: number;
  ablation: AblationCode;
  model_key: string;
  experiment_id: string;
}

export interface ExperimentQueryParams {
  model_key?: string;
  experiment_id?: string;
  ablation?: AblationCode;
}

export interface SampleListQueryParams extends ExperimentQueryParams {
  page?: number;
  page_size?: number;
  decision?: VerificationDecision;
  matched_gt?: boolean;
  gt_label?: GroundTruthLabel;
}
