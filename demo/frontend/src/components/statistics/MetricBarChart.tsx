import { ChartCard } from "@/components/statistics/ChartCard";
import type { MetricKey, PrimaryMetrics } from "@/components/statistics/types";

const METRIC_ORDER: MetricKey[] = [
  "precision",
  "recall",
  "specificity",
  "f1",
  "accuracy",
  "balanced_accuracy",
  "uncertain_rate",
  "coverage",
];

const METRIC_LABELS: Record<MetricKey, string> = {
  precision: "Precision",
  recall: "Recall",
  specificity: "Specificity",
  f1: "F1",
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced Acc.",
  uncertain_rate: "Uncertain Rate",
  coverage: "Coverage",
};

export interface MetricBarChartProps {
  metrics: PrimaryMetrics;
  title?: string;
  subtitle?: string;
}

function formatMetric(value: number | null): string {
  if (value === null) return "—";
  return value.toFixed(3);
}

/**
 * Protocol metrics table derived from evaluation confusion / decision counts.
 */
export function MetricBarChart({
  metrics,
  title = "Verification protocol metrics",
  subtitle = "Definitive decisions only for P/R/Spec/F1/Acc/BA; Uncertain rate & Coverage use full dataset",
}: MetricBarChartProps) {
  return (
    <ChartCard title={title} subtitle={subtitle}>
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
        {METRIC_ORDER.map((key) => {
          const value = metrics[key];
          return (
            <div key={key} className="border-b border-slate-100 py-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                {METRIC_LABELS[key]}
              </p>
              <p className="mt-1 font-mono text-lg text-slate-900">{formatMetric(value)}</p>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}
