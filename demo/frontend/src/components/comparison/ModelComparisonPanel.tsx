import { ModelComparisonCard } from "@/components/comparison/ModelComparisonCard";
import type { SampleComparisonData } from "@/components/comparison/types";

export interface ModelComparisonPanelProps {
  data: SampleComparisonData;
  /** modelKey to visually emphasize (optional). */
  activeModelKey?: string;
  className?: string;
}

/**
 * Side-by-side comparison of VLM predictions for one verification sample.
 */
export function ModelComparisonPanel({
  data,
  activeModelKey,
  className = "",
}: ModelComparisonPanelProps) {
  return (
    <section
      className={`flex flex-col overflow-hidden border-t border-slate-700/60 bg-slate-950/50 ${className}`}
      aria-label="Model comparison"
    >
      <header className="flex shrink-0 items-center justify-between border-b border-slate-700/60 px-4 py-2.5">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Model comparison
          </h2>
          <p className="font-mono text-sm text-slate-200">{data.sampleId}</p>
        </div>
        <p className="truncate font-mono text-[11px] text-slate-500">{data.imageName}</p>
      </header>

      <div className="min-h-0 flex-1 overflow-x-auto overflow-y-hidden p-4">
        <div className="flex gap-4">
          {data.predictions.map((prediction) => (
            <ModelComparisonCard
              key={prediction.modelKey}
              prediction={prediction}
              highlighted={activeModelKey === prediction.modelKey}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
