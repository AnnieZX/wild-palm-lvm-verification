import { ComparisonField } from "@/components/comparison/ComparisonField";
import { DecisionBadge } from "@/components/comparison/DecisionBadge";
import { GroundTruthBlock } from "@/components/comparison/GroundTruthBlock";
import type { ModelPredictionResult } from "@/components/comparison/types";

export interface ModelComparisonCardProps {
  prediction: ModelPredictionResult;
  /** Highlight when this model matches the active dashboard selection. */
  highlighted?: boolean;
}

function formatProcessingTime(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

function formatConfidence(confidence: number | null): string {
  if (confidence === null) return "—";
  return `${(confidence * 100).toFixed(1)}%`;
}

/**
 * Single-model prediction card for side-by-side comparison.
 */
export function ModelComparisonCard({ prediction, highlighted = false }: ModelComparisonCardProps) {
  return (
    <article
      className={`flex w-[260px] shrink-0 flex-col rounded-lg border bg-slate-900/80 ${
        highlighted
          ? "wp-badge-selection border-selection-500/60 ring-1 ring-selection-500/30"
          : "border-slate-700/80"
      }`}
      aria-label={`${prediction.modelName} prediction`}
    >
      <header className="border-b border-slate-700/60 px-4 py-3">
        <h3 className="text-sm font-semibold text-slate-100">{prediction.modelName}</h3>
        <p className="font-mono text-[10px] text-slate-500">{prediction.modelKey}</p>
      </header>

      <div className="flex flex-1 flex-col gap-4 p-4">
        <ComparisonField label="Decision">
          <DecisionBadge decision={prediction.decision} size="sm" />
        </ComparisonField>

        <ComparisonField label="Confidence">
          <span className="font-mono text-base font-semibold text-slate-100">
            {formatConfidence(prediction.confidence)}
          </span>
        </ComparisonField>

        <ComparisonField label="Reasoning">
          <p>{prediction.reasoning}</p>
          {prediction.visualReasoning ? (
            <p className="mt-2 text-slate-400">{prediction.visualReasoning}</p>
          ) : null}
        </ComparisonField>

        <ComparisonField label="Processing time" mono>
          {formatProcessingTime(prediction.processingTimeMs)}
        </ComparisonField>

        <ComparisonField label="Prompt version" mono>
          {prediction.promptVersion}
        </ComparisonField>

        <ComparisonField label="Ground truth">
          <GroundTruthBlock groundTruth={prediction.groundTruth} compact />
        </ComparisonField>
      </div>
    </article>
  );
}
