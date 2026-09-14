/** Frontend statistics dashboard view models (API-backed). */

export interface PrimaryMetrics {
  precision: number | null;
  recall: number | null;
  specificity: number | null;
  f1: number | null;
  accuracy: number | null;
  balanced_accuracy: number | null;
  uncertain_rate: number | null;
  coverage: number | null;
}

export interface DecisionDistributionCounts {
  Reliable: number;
  Uncertain: number;
  Unreliable: number;
}

export interface ConfusionMatrixData {
  row_labels: string[];
  col_labels: string[];
  cells: number[][];
  uncertain_excluded: number;
}

export interface ModelComparisonRow {
  model_key: string;
  display_name: string;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  specificity?: number | null;
  balanced_accuracy?: number | null;
}

export interface PromptComparisonRow {
  ablation: string;
  condition: string;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  specificity?: number | null;
  balanced_accuracy?: number | null;
}

export interface StatisticsDashboardData {
  experiment_id: string;
  model_key: string;
  dataset_size: number;
  evaluated_samples: number;
  primary_metrics: PrimaryMetrics;
  decision_distribution: DecisionDistributionCounts;
  confusion_matrix: ConfusionMatrixData;
  model_comparison: ModelComparisonRow[];
  prompt_comparison: PromptComparisonRow[];
}

export type MetricKey = keyof PrimaryMetrics;

export interface MetricSeries {
  label: string;
  value: number;
}
