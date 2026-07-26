import type { SampleDetail } from "@shared/types";
import type { ModelInfo } from "@shared/types";

import { GroundTruthCard } from "@/components/info-panel/GroundTruthCard";
import { ModelMetadataCard } from "@/components/info-panel/ModelMetadataCard";
import { PredictionCard } from "@/components/info-panel/PredictionCard";
import { ReasoningCard } from "@/components/info-panel/ReasoningCard";
import { YoloConfidenceCard } from "@/components/info-panel/YoloConfidenceCard";

interface InformationPanelProps {
  sample: SampleDetail;
  model: ModelInfo;
  experimentId: string;
  ablation: string;
}

export function InformationPanel({
  sample,
  model,
  experimentId,
  ablation,
}: InformationPanelProps) {
  return (
    <aside
      className="flex h-full flex-col overflow-hidden border-t border-slate-200/80 bg-slate-50/50 xl:border-l xl:border-t-0"
      aria-label="Sample information"
    >
      <div className="border-b border-slate-200/80 bg-white px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-800">Information</h2>
        <p className="text-xs text-slate-500">Detection &amp; verification metadata</p>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-3">
        <YoloConfidenceCard sample={sample} />
        <GroundTruthCard sample={sample} />
        <PredictionCard sample={sample} />
        <ReasoningCard sample={sample} />
        <ModelMetadataCard model={model} experimentId={experimentId} ablation={ablation} />
      </div>
    </aside>
  );
}
