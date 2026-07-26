import type { SampleDetail } from "@shared/types";

import { InfoCard } from "@/components/info-panel/InfoCard";

interface GroundTruthCardProps {
  sample: SampleDetail;
}

function formatBBox(bbox: SampleDetail["gt_bbox"]): string {
  if (!bbox) return "—";
  return `[${bbox.x}, ${bbox.y}, ${bbox.width}×${bbox.height}]`;
}

export function GroundTruthCard({ sample }: GroundTruthCardProps) {
  return (
    <InfoCard title="Ground Truth" subtitle="LabelMe palm annotation">
      <dl className="space-y-2">
        <div className="flex justify-between gap-4">
          <dt className="text-slate-500">GT match</dt>
          <dd className="font-medium text-slate-900">
            {sample.matched_gt ? "Matched" : "Unmatched"}
          </dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-slate-500">Max IoU</dt>
          <dd className="font-mono font-medium text-slate-900">
            {sample.max_iou.toFixed(2)}
          </dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-slate-500">GT label</dt>
          <dd className="font-medium capitalize text-slate-900">{sample.gt_label}</dd>
        </div>
        <div>
          <dt className="text-slate-500">GT bbox</dt>
          <dd className="mt-1 font-mono text-xs text-slate-700">{formatBBox(sample.gt_bbox)}</dd>
        </div>
      </dl>
    </InfoCard>
  );
}
