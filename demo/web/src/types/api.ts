export type Condition = "A1" | "A2" | "A3" | "A4" | "A5";
export type VlmLabel = "Reliable" | "Uncertain" | "Unreliable";
export type SemanticLabel = "palm" | "ambiguous" | "non_palm";
export type VisitorLabel = SemanticLabel | "skip";
export type Quintile = "Q1" | "Q2" | "Q3" | "Q4" | "Q5";
export type Bbox = [number, number, number, number];

export type CellStatus = "full_clean" | "partial" | "incomplete" | "not_run" | "not_audited";
export type Degeneracy = "single_label" | "near_collapse_all" | "near_collapse_some" | null;

export interface Summary {
  n_detections: number;
  n_images: number;
  conditions: Condition[];
  yolo_confidence_min: number;
  n_models_registered: number;
  n_models_with_any_usable_condition: number;
  n_models_all_five_usable: number;
  n_usable_cells: number;
  n_decisions: number;
  labelme_matched: number;
  labelme_unmatched: number;
  semantic_review: Partial<Record<SemanticLabel, number>>;
  iou_threshold: number;
  audit_status_counts: Record<string, number> | null;
  sources: Record<string, string>;
}

export interface ModelCondition {
  status: CellStatus;
  files_present: number | null;
  n_valid: number | null;
}

export interface ModelInfo {
  model_key: string;
  display: string;
  family: string;
  nominal_size: string | null;
  nominal_b: number | null;
  params_billions: number | null;
  registry_order: number;
  run_dir: string | null;
  expected_status: string | null;
  condition_policy: string | null;
  in_core_panel: boolean;
  ladder: string | null;
  conditions: Record<Condition, ModelCondition>;
  n_usable_conditions: number;
  label_usage: {
    pattern: string | null;
    labels_never_used: string | null;
    n_single_label: number | null;
    n_majority_ge_0_99: number | null;
    majority_share_max: number | null;
    entropy_min: number | null;
    entropy_max: number | null;
  } | null;
  degenerate: Degeneracy;
}

export interface DetectionSummary {
  detection_id: string;
  image_id: string;
  bbox_xywh: Bbox;
  bbox_area: number;
  norm_area: number;
  image_width: number;
  image_height: number;
  yolo_confidence: number;
  confidence_quintile: Quintile;
  area_quintile: Quintile;
  labelme_matched: boolean | null;
  max_iou: number | null;
  semantic_label: SemanticLabel | null;
  unmatched_reason: string | null;
}

export interface DetectionList {
  total: number;
  page: number;
  page_size: number;
  items: DetectionSummary[];
}

export interface DetectionDetail extends DetectionSummary {
  detections_in_image: number;
  other_detections: { detection_id: string; bbox_xywh: Bbox; yolo_confidence: number }[];
  labelme: { available: boolean; palm_boxes_xywh: Bbox[] };
  condition_inputs: Record<Condition, { image_url: string; prompt_url: string }>;
  raw_image_url: string;
}

export type CellState = "decision" | "parse_error" | "missing" | "pending" | "not_run" | "not_audited";

export interface PredictionCell {
  state: CellState;
  decision: VlmLabel | null;
  cell_status: CellStatus;
}

export interface PredictionRow {
  model_key: string;
  display: string;
  family: string;
  nominal_size: string | null;
  params_billions: number | null;
  degenerate: Degeneracy;
  cells: Record<Condition, PredictionCell>;
  changes_across_conditions: boolean;
}

export interface Predictions {
  detection_id: string;
  conditions: Condition[];
  models: PredictionRow[];
}

export interface Reasoning {
  available: boolean;
  reason?: string;
  source?: string;
  decision?: VlmLabel | null;
  analysis_table_decision?: VlmLabel | null;
  consistent_with_analysis_table?: boolean;
  visual_reasoning?: string;
  confidence_reasoning?: string;
  parse_error?: string;
  inference_error?: string;
  raw_response?: string;
  raw_response_truncated?: boolean;
  runtime_seconds?: number;
  model_name?: string;
  experiment_id?: string;
}

export interface PromptInfo {
  condition: Condition;
  condition_dir: string;
  source: string;
  input_image_section: string | null;
  metadata_section: string | null;
  full_text: string;
}

export interface CuratedCase {
  category: string;
  rank_in_category: number;
  sample_id: string;
  matched_gt: boolean;
  yolo_confidence: number;
  semantic_label: string;
}

export interface ConditionPair {
  pair: string;
  group: string;
  note: string | null;
}

export interface TransitionCell {
  from: VlmLabel;
  to: VlmLabel;
  count: number;
  share_of_n?: number;
  share?: number;
}

export type Distribution = Partial<Record<VlmLabel, { count: number; rate: number }>>;

export interface ContextShiftModel {
  model: string;
  model_key: string | null;
  group: string;
  pair: string;
  n: number;
  pair_status: string;
  n_changed: number;
  changed_rate: number;
  toward_rejection_rate: number;
  toward_acceptance_rate: number;
  into_U_rate: number | null;
  into_Ur_rate: number | null;
  into_R_rate: number | null;
  matrix: TransitionCell[];
  distributions: Record<string, Distribution>;
  [k: string]: unknown;
}

export interface ContextShift {
  available: boolean;
  pair: string;
  pairs: ConditionPair[];
  models: ContextShiftModel[];
  sources: string[];
}

export interface LabelUsageCell {
  condition: Condition;
  condition_status: string;
  n_valid: number;
  share_R: number;
  share_U: number;
  share_Ur: number;
  entropy_bits: number;
  majority_label: VlmLabel;
  majority_share: number;
  diag_single_label: boolean;
  "diag_majority_ge_0.99": boolean;
}

export interface LadderCheckpoint {
  ladder: string;
  rank_in_ladder: number;
  model_key: string;
  model: string;
  nominal_b: number | null;
  params_billions_counted: number | null;
  status: string;
  single_label_A1_and_A5: boolean | null;
  n_paired_A1_A5: number | null;
  A1_R: number | null; A1_U: number | null; A1_Ur: number | null; A1_entropy: number | null;
  A5_R: number | null; A5_U: number | null; A5_Ur: number | null; A5_entropy: number | null;
  A1_majority_share: number | null; A5_majority_share: number | null;
  changed: number | null;
  toward_rejection: number | null;
  toward_acceptance: number | null;
  into_U: number | null;
  into_Ur: number | null;
  changed_ci_low: number | null; changed_ci_high: number | null;
  into_U_ci_low: number | null; into_U_ci_high: number | null;
  into_Ur_ci_low: number | null; into_Ur_ci_high: number | null;
  verdict_confidence: string | null;
  verdict_box_area: string | null;
  note: string | null;
  matrix?: TransitionCell[];
  label_usage?: LabelUsageCell[];
}

export interface LadderOrdering {
  ladder: string;
  quantity: string;
  checkpoints_in_size_order: string;
  values: string;
  excluded_single_label: string | null;
  not_available: string | null;
  observed_pattern: string;
  note: string;
}

export interface Scaling {
  available: boolean;
  ladders: { ladder: string; checkpoints: LadderCheckpoint[]; orderings: LadderOrdering[] }[];
  sources: string[];
}

export interface QuintileRate {
  property: "confidence" | "box_area";
  quintile: Quintile;
  event: string;
  n_at_risk: number;
  n_event: number;
  rate: number;
  ci_low: number | null;
  ci_high: number | null;
}

export interface QuintileContrast {
  property: "confidence" | "box_area";
  event: string;
  n_paired: number;
  rate_Q1: number;
  rate_Q5: number;
  diff_Q1_minus_Q5: number;
  ci_low: number | null;
  ci_high: number | null;
  verdict: string;
  single_label_both: boolean;
}

export interface DifficultyModel {
  model: string;
  model_key: string | null;
  rates: QuintileRate[];
  contrasts: QuintileContrast[];
}

export interface Difficulty {
  available: boolean;
  pair: string;
  pairs: string[];
  models: DifficultyModel[];
  hypothesis_summary: {
    pair: string; property: string; n_models: number; n_evaluable: number; n_supports: number;
    n_reverses: number; n_null: number; supports: string | null; reverses: string | null;
    null: string | null; not_evaluable: string | null;
  }[];
  property_info: {
    spearman_confidence_vs_norm_area: number;
    n: number;
    confidence_quintile_edges: string;
    norm_area_quintile_edges: string;
  } | null;
  confidence_x_area: { model: string; pair: string; confidence_half: string; area_half: string; n: number; changed_rate: number }[];
  sources: string[];
}

export interface SemanticReview {
  available: boolean;
  unmatched_review: {
    n: number;
    label_counts: Record<SemanticLabel, number>;
    reviewers: string[];
    review_complete: boolean;
    labelme_matched_reviewed: number;
    first_timestamp: string;
    last_timestamp: string;
  };
  pilot: { rows: number; min_conf: number; max_conf: number; inside_5747_cohort: boolean; has_vlm_predictions: boolean } | null;
  iou_status_by_semantic: { iou_status: string; n: number; reviewed: number; palm: string | number; ambiguous: string | number; non_palm: string | number }[];
  unmatched_reason_by_semantic: { unmatched_reason_geometric: string; palm: number; ambiguous: number; non_palm: number; total: number }[];
  vlm_verdicts_by_semantic_label: { model: string; condition: Condition; semantic_label: SemanticLabel; n: number; n_R: number; n_U: number; n_Ur: number; rate_R: number; rate_U: number; rate_Ur: number }[];
  alignment_cells_semantic_composition: { model: string; condition: Condition; alignment_cell: string; verdict: VlmLabel; n: number; n_palm: number; n_ambiguous: number; n_non_palm: number }[];
  pilot_by_confidence_bin: { confidence_bin: string; palm: number; ambiguous: number; non_palm: number; n: number; bin_population: number }[];
  subsets: { subset: string; n: number; palm: number; ambiguous: number; non_palm: number }[];
  alignment_metrics: {
    model_key: string; model: string; condition: Condition; condition_status: string;
    align_R_unmatched: number; align_Ur_unmatched: number; align_U_unmatched: number;
    n_decided_unmatched: number; align_specificity: number | null;
  }[];
  sources: string[];
}

export interface ReviewItemPublic {
  index: number;
  image_url: string;
  bbox_xywh: Bbox;
  image_width: number | null;
  image_height: number | null;
  visitor_label: VisitorLabel | null;
}

export interface DeckInfo {
  title: string;
  description: string;
  source: string;
  pool_size?: number;
}

export interface ReviewSession {
  session_id: string;
  deck: string;
  deck_info: DeckInfo;
  question: string;
  labels: VisitorLabel[];
  definitions: Record<SemanticLabel, string>;
  items: ReviewItemPublic[];
  n_answered: number;
  complete: boolean;
}

export interface ReviewDecks {
  question: string;
  labels: VisitorLabel[];
  definitions: Record<SemanticLabel, string>;
  decks: Record<string, DeckInfo>;
}

export interface ReviewResultItem {
  index: number;
  ref_id: string;
  image_id: string;
  image_url: string;
  bbox_xywh: Bbox;
  image_width: number | null;
  image_height: number | null;
  visitor_label: VisitorLabel;
  research_label: SemanticLabel;
  agrees: boolean | null;
  labelme_palm_boxes_xywh: Bbox[];
  detection_id?: string;
  yolo_confidence?: number;
  max_iou?: number;
  labelme_matched?: boolean;
  unmatched_reason?: string | null;
  confidence_bin?: string;
  vlm?: Record<string, Record<Condition, VlmLabel | null>> | null;
}

export interface ReviewResults {
  session_id: string;
  deck: string;
  deck_info: DeckInfo;
  items: ReviewResultItem[];
  n_answered: number;
  n_agree: number;
  models: { model_key: string; display: string; family: string; degenerate: Degeneracy; usable_conditions: Condition[] }[];
}
