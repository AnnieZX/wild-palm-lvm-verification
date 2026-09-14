import { ChartCard } from "@/components/statistics/ChartCard";

export interface ComparisonRow {
  id: string;
  label: string;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
}

type HighlightMetric = "precision" | "recall" | "f1" | "accuracy";

export interface GroupedComparisonChartProps {
  title: string;
  subtitle?: string;
  rows: ComparisonRow[];
  highlightMetric?: HighlightMetric;
}

const METRIC_COLORS: Record<HighlightMetric, string> = {
  precision: "bg-selection-400",
  recall: "bg-forest-400",
  f1: "bg-forest-600",
  accuracy: "bg-forest-500",
};

/**
 * Compact F1 comparison bars for models or ablations with available metrics.
 */
export function GroupedComparisonChart({
  title,
  subtitle,
  rows,
  highlightMetric = "f1",
}: GroupedComparisonChartProps) {
  if (rows.length === 0) {
    return (
      <ChartCard title={title} subtitle={subtitle}>
        <p className="text-xs text-slate-500">No comparison rows available.</p>
      </ChartCard>
    );
  }

  const maxValue = Math.max(...rows.map((row) => row[highlightMetric]), 0.01);

  return (
    <ChartCard title={title} subtitle={subtitle}>
      <div className="flex flex-1 flex-col gap-3 overflow-x-auto">
        {rows.map((row) => {
          const value = row[highlightMetric];
          const widthPct = (value / maxValue) * 100;
          return (
            <div key={row.id} className="min-w-[240px]">
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="truncate text-xs font-medium text-slate-700">{row.label}</span>
                <span className="shrink-0 font-mono text-xs text-slate-600">
                  {highlightMetric.toUpperCase()} {value.toFixed(3)}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded bg-slate-100">
                <div
                  className={`h-full rounded ${METRIC_COLORS[highlightMetric]}`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
              <dl className="mt-1 grid grid-cols-4 gap-1 font-mono text-[10px] text-slate-500">
                <div>P {row.precision.toFixed(2)}</div>
                <div>R {row.recall.toFixed(2)}</div>
                <div>F1 {row.f1.toFixed(2)}</div>
                <div>Acc {row.accuracy.toFixed(2)}</div>
              </dl>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}
