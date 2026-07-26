import { ChartCard } from "@/components/statistics/ChartCard";
import type { MetricKey, PrimaryMetrics } from "@/components/statistics/types";

const METRIC_LABELS: Record<MetricKey, string> = {
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  accuracy: "Accuracy",
};

const METRIC_COLORS: Record<MetricKey, string> = {
  precision: "bg-selection-500",
  recall: "bg-forest-500",
  f1: "bg-forest-700",
  accuracy: "bg-forest-400",
};

export interface MetricBarChartProps {
  metrics: PrimaryMetrics;
  title?: string;
  subtitle?: string;
}

/**
 * Placeholder bar chart for binary verification metrics (0–1 scale).
 */
export function MetricBarChart({
  metrics,
  title = "Verification metrics",
  subtitle = "Definitive decisions only (Uncertain excluded)",
}: MetricBarChartProps) {
  const entries = (Object.keys(METRIC_LABELS) as MetricKey[]).map((key) => ({
    key,
    label: METRIC_LABELS[key],
    value: metrics[key],
    color: METRIC_COLORS[key],
  }));

  return (
    <ChartCard title={title} subtitle={subtitle}>
      <div className="flex flex-1 items-end justify-around gap-3 pt-4">
        {entries.map((entry) => {
          const pct = Math.round(entry.value * 100);
          return (
            <div key={entry.key} className="flex flex-1 flex-col items-center gap-2">
              <span className="font-mono text-sm font-semibold text-slate-800">
                {entry.value.toFixed(3)}
              </span>
              <div className="flex h-36 w-full max-w-[72px] items-end rounded-t-md bg-slate-100">
                <div
                  className={`w-full rounded-t-md ${entry.color} transition-all`}
                  style={{ height: `${pct}%` }}
                  role="img"
                  aria-label={`${entry.label} ${pct}%`}
                />
              </div>
              <span className="text-center text-xs font-medium text-slate-600">
                {entry.label}
              </span>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}
