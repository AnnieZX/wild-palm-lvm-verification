import type { SampleDetail } from "@shared/types";

import { InfoCard } from "@/components/info-panel/InfoCard";

interface YoloConfidenceCardProps {
  sample: SampleDetail;
}

export function YoloConfidenceCard({ sample }: YoloConfidenceCardProps) {
  const confidence = sample.yolo_confidence ?? 0;
  const pct = Math.round(confidence * 100);

  return (
    <InfoCard title="YOLO Confidence" subtitle="Detector score (auxiliary)">
      <div className="space-y-3">
        <div className="flex items-baseline justify-between">
          <span className="font-mono text-2xl font-semibold text-slate-900">
            {confidence.toFixed(2)}
          </span>
          <span className="text-xs text-slate-500">{pct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-forest-600"
            style={{ width: `${pct}%` }}
            aria-hidden
          />
        </div>
        <p className="text-xs leading-relaxed text-slate-500">
          Placeholder — confidence bar for layout only.
        </p>
      </div>
    </InfoCard>
  );
}
