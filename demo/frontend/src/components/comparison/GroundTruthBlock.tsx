import type { ComparisonGroundTruth } from "@/components/comparison/types";

export interface GroundTruthBlockProps {
  groundTruth: ComparisonGroundTruth;
  compact?: boolean;
}

export function GroundTruthBlock({ groundTruth, compact = false }: GroundTruthBlockProps) {
  return (
    <dl className={`grid gap-2 ${compact ? "grid-cols-2" : "grid-cols-1"}`}>
      <div className="flex justify-between gap-2">
        <dt className="text-slate-500">Match</dt>
        <dd className="font-medium text-slate-200">
          {groundTruth.matched ? "Matched" : "Unmatched"}
        </dd>
      </div>
      <div className="flex justify-between gap-2">
        <dt className="text-slate-500">Label</dt>
        <dd className="capitalize text-slate-200">{groundTruth.label}</dd>
      </div>
      <div className="flex justify-between gap-2">
        <dt className="text-slate-500">Max IoU</dt>
        <dd className="font-mono text-slate-200">{groundTruth.maxIou.toFixed(2)}</dd>
      </div>
    </dl>
  );
}
