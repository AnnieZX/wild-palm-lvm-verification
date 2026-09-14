import { DecisionBadge } from "@/components/comparison/DecisionBadge";
import type { SampleComparisonData } from "@/components/comparison/types";

export interface ModelComparisonPanelProps {
  data: SampleComparisonData;
  activeModelKey?: string;
  className?: string;
}

/**
 * Same-sample model comparison as a compact analysis table.
 */
export function ModelComparisonPanel({
  data,
  activeModelKey,
  className = "",
}: ModelComparisonPanelProps) {
  return (
    <section
      className={`flex flex-col overflow-hidden border-t border-slate-700/70 bg-slate-950 ${className}`}
      aria-label="Model comparison"
    >
      <header className="flex shrink-0 items-center justify-between border-b border-slate-800 px-3 py-1.5">
        <h2 className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          Model comparison
        </h2>
        <p className="font-mono text-[11px] text-slate-500">{data.sampleId}</p>
      </header>

      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full border-collapse text-left text-xs">
          <thead className="sticky top-0 bg-slate-950 text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-slate-800">
              <th className="px-3 py-1.5 font-medium">Model</th>
              <th className="px-3 py-1.5 font-medium">Decision</th>
              <th className="px-3 py-1.5 font-medium">Status</th>
              <th className="px-3 py-1.5 font-medium">Ablation</th>
              <th className="px-3 py-1.5 font-medium">GT / IoU</th>
            </tr>
          </thead>
          <tbody>
            {data.predictions.map((prediction) => {
              const active = activeModelKey === prediction.modelKey;
              const status =
                prediction.decision === "Reliable"
                  ? "✓"
                  : prediction.decision === "Unreliable"
                    ? "✕"
                    : "·";
              return (
                <tr
                  key={prediction.modelKey}
                  className={`border-b border-slate-800/80 ${
                    active ? "bg-slate-900" : "bg-transparent"
                  }`}
                >
                  <td className="px-3 py-2">
                    <div className="text-slate-100">{prediction.modelName}</div>
                    <div className="font-mono text-[10px] text-slate-500">
                      {prediction.modelKey}
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <DecisionBadge decision={prediction.decision} size="sm" />
                  </td>
                  <td className="px-3 py-2 font-mono text-slate-300">{status}</td>
                  <td className="px-3 py-2 font-mono text-slate-400">
                    {prediction.promptVersion}
                  </td>
                  <td className="px-3 py-2 font-mono text-slate-400">
                    {prediction.groundTruth.matched ? "+" : "−"} /{" "}
                    {prediction.groundTruth.maxIou.toFixed(3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
