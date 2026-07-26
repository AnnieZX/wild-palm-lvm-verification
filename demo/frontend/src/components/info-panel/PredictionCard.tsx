import type { SampleDetail, VerificationDecision } from "@shared/types";

import { InfoCard } from "@/components/info-panel/InfoCard";
import { decisionBadgeLight } from "@/theme/decision-styles";

interface PredictionCardProps {
  sample: SampleDetail;
}

export function PredictionCard({ sample }: PredictionCardProps) {
  const decision = sample.decision ?? "Uncertain";

  return (
    <InfoCard title="Prediction" subtitle="VLM verification decision">
      <div className="space-y-3">
        <span
          className={`inline-flex rounded-md px-2.5 py-1 text-sm font-semibold ring-1 ring-inset ${decisionBadgeLight[decision as VerificationDecision]}`}
        >
          {decision}
        </span>
        <dl className="space-y-1 text-xs">
          <div className="flex justify-between gap-2">
            <dt className="text-slate-500">Sample ID</dt>
            <dd className="font-mono text-slate-800">{sample.sample_id}</dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt className="text-slate-500">Image</dt>
            <dd className="truncate font-mono text-slate-800">{sample.image_name}</dd>
          </div>
        </dl>
      </div>
    </InfoCard>
  );
}
