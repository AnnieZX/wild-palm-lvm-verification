import type { SampleDetail } from "@shared/types";

import { InfoCard } from "@/components/info-panel/InfoCard";

interface YoloConfidenceCardProps {
  sample: SampleDetail;
}

export function YoloConfidenceCard({ sample }: YoloConfidenceCardProps) {
  const confidence = sample.yolo_confidence;

  return (
    <InfoCard title="YOLO Confidence">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-xl font-semibold text-slate-900">
          {confidence === null ? "—" : confidence.toFixed(3)}
        </span>
      </div>
    </InfoCard>
  );
}
