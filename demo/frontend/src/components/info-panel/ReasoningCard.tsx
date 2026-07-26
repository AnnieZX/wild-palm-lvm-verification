import type { SampleDetail } from "@shared/types";

import { InfoCard } from "@/components/info-panel/InfoCard";

interface ReasoningCardProps {
  sample: SampleDetail;
}

export function ReasoningCard({ sample }: ReasoningCardProps) {
  return (
    <InfoCard title="Reasoning" subtitle="Model-generated explanations">
      <div className="space-y-4">
        <div>
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-400">
            Visual
          </p>
          <p className="text-xs leading-relaxed text-slate-600">
            {sample.visual_reasoning || "—"}
          </p>
        </div>
        <div>
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-400">
            Confidence
          </p>
          <p className="text-xs leading-relaxed text-slate-600">
            {sample.confidence_reasoning || "—"}
          </p>
        </div>
      </div>
    </InfoCard>
  );
}
