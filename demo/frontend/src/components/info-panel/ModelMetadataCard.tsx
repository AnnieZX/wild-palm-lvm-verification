import type { ModelInfo } from "@shared/types";

import { InfoCard } from "@/components/info-panel/InfoCard";

interface ModelMetadataCardProps {
  model: ModelInfo;
  experimentId: string;
  ablation: string;
}

export function ModelMetadataCard({ model, experimentId, ablation }: ModelMetadataCardProps) {
  const experiment = model.experiments.find((item) => item.experiment_id === experimentId);

  return (
    <InfoCard title="Model Metadata" subtitle="Experiment context">
      <dl className="space-y-2 text-xs">
        <div>
          <dt className="text-slate-500">Model</dt>
          <dd className="font-medium text-slate-900">{model.display_name}</dd>
          <dd className="font-mono text-[11px] text-slate-500">{model.model_key}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Experiment</dt>
          <dd className="font-mono text-slate-800">{experimentId}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Ablation</dt>
          <dd className="font-mono text-slate-800">{ablation}</dd>
        </div>
        {experiment ? (
          <div>
            <dt className="text-slate-500">Dataset size</dt>
            <dd className="font-medium text-slate-900">
              {experiment.sample_count.toLocaleString()} samples
            </dd>
          </div>
        ) : null}
      </dl>
    </InfoCard>
  );
}
