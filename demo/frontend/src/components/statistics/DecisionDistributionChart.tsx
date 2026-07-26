import { ChartCard } from "@/components/statistics/ChartCard";
import { decisionChartSegment } from "@/theme/decision-styles";
import type { DecisionDistributionCounts } from "@/components/statistics/types";

export interface DecisionDistributionChartProps {
  distribution: DecisionDistributionCounts;
}

/**
 * Placeholder stacked bar for Reliable / Uncertain / Unreliable counts.
 */
export function DecisionDistributionChart({ distribution }: DecisionDistributionChartProps) {
  const segments = (
    [
      ["Reliable", distribution.Reliable],
      ["Uncertain", distribution.Uncertain],
      ["Unreliable", distribution.Unreliable],
    ] as const
  ).filter(([, count]) => count > 0);

  const total = segments.reduce((sum, [, count]) => sum + count, 0);

  return (
    <ChartCard
      title="Decision distribution"
      subtitle={`${total.toLocaleString()} evaluated samples`}
    >
      <div className="flex flex-1 flex-col justify-center gap-4">
        <div className="flex h-10 w-full overflow-hidden rounded-lg">
          {segments.map(([label, count]) => {
            const widthPct = (count / total) * 100;
            return (
              <div
                key={label}
                className={`${decisionChartSegment[label]} flex items-center justify-center text-[10px] font-semibold text-white`}
                style={{ width: `${widthPct}%` }}
                title={`${label}: ${count}`}
              >
                {widthPct > 12 ? `${Math.round(widthPct)}%` : ""}
              </div>
            );
          })}
        </div>

        <ul className="grid grid-cols-3 gap-2 text-xs">
          {segments.map(([label, count]) => (
            <li key={label} className="flex items-center gap-2">
              <span
                className={`h-2.5 w-2.5 shrink-0 rounded-sm ${decisionChartSegment[label]}`}
                aria-hidden
              />
              <span className="text-slate-600">{label}</span>
              <span className="ml-auto font-mono text-slate-800">{count}</span>
            </li>
          ))}
        </ul>
      </div>
    </ChartCard>
  );
}
