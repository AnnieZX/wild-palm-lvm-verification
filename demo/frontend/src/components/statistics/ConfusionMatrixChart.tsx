import { Fragment } from "react";

import { ChartCard } from "@/components/statistics/ChartCard";
import type { ConfusionMatrixData } from "@/components/statistics/types";

function cellIntensity(value: number, max: number): string {
  const t = max > 0 ? value / max : 0;
  const alpha = 0.12 + t * 0.55;
  return `rgba(50 80 60 / ${alpha})`;
}

export interface ConfusionMatrixChartProps {
  matrix: ConfusionMatrixData;
}

/**
 * Placeholder heatmap grid for TP/FP/TN/FN counts.
 */
export function ConfusionMatrixChart({ matrix }: ConfusionMatrixChartProps) {
  const flat = matrix.cells.flat();
  const maxCell = Math.max(...flat, 1);

  return (
    <ChartCard
      title="Confusion matrix"
      subtitle={`Uncertain excluded: ${matrix.uncertain_excluded}`}
    >
      <div className="flex flex-1 flex-col justify-center">
        <div
          className="grid gap-1"
          style={{
            gridTemplateColumns: `auto repeat(${matrix.col_labels.length}, 1fr)`,
          }}
        >
          <div />
          {matrix.col_labels.map((label) => (
            <div
              key={label}
              className="px-1 py-2 text-center text-[10px] font-medium text-slate-500"
            >
              {label}
            </div>
          ))}

          {matrix.cells.map((row, rowIndex) => (
            <Fragment key={matrix.row_labels[rowIndex]}>
              <div className="flex items-center pr-2 text-[10px] font-medium text-slate-500">
                {matrix.row_labels[rowIndex]}
              </div>
              {row.map((value, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="flex aspect-square min-h-[64px] items-center justify-center rounded-md border border-slate-200 font-mono text-sm font-semibold text-slate-800"
                  style={{ backgroundColor: cellIntensity(value, maxCell) }}
                >
                  {value}
                </div>
              ))}
            </Fragment>
          ))}
        </div>
      </div>
    </ChartCard>
  );
}
