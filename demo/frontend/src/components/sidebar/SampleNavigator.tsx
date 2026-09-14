"use client";

import type { SampleSummary } from "@shared/types";

interface SampleNavigatorProps {
  samples: SampleSummary[];
  selectedSampleId: string | null;
  total: number;
  onSelect: (sampleId: string) => void;
  onPrevious: () => void;
  onNext: () => void;
  canPrevious: boolean;
  canNext: boolean;
}

function shortId(sampleId: string): string {
  return sampleId.replace(/^sample_/, "");
}

export function SampleNavigator({
  samples,
  selectedSampleId,
  total,
  onSelect,
  onPrevious,
  onNext,
  canPrevious,
  canNext,
}: SampleNavigatorProps) {
  const activeIndex = samples.findIndex((item) => item.sample_id === selectedSampleId);

  return (
    <section className="flex min-h-0 flex-1 flex-col border-t border-slate-200" aria-label="Results">
      <div className="flex items-center justify-between gap-2 border-b border-slate-200 px-3 py-2">
        <div>
          <p className="wp-label">Results</p>
          <p className="font-mono text-[11px] text-slate-600">
            {samples.length.toLocaleString()} listed
            {total !== samples.length ? ` · ${total.toLocaleString()} filtered` : ""}
            {activeIndex >= 0 ? ` · ${activeIndex + 1}/${samples.length}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            className="rounded border border-slate-200 px-2 py-0.5 font-mono text-[11px] text-slate-700 disabled:opacity-40"
            onClick={onPrevious}
            disabled={!canPrevious}
            aria-label="Previous sample"
          >
            ↑
          </button>
          <button
            type="button"
            className="rounded border border-slate-200 px-2 py-0.5 font-mono text-[11px] text-slate-700 disabled:opacity-40"
            onClick={onNext}
            disabled={!canNext}
            aria-label="Next sample"
          >
            ↓
          </button>
        </div>
      </div>

      <ul className="min-h-0 flex-1 overflow-y-auto">
        {samples.length === 0 ? (
          <li className="px-3 py-4 text-xs text-slate-500">No samples match filters.</li>
        ) : (
          samples.map((sample) => {
            const active = sample.sample_id === selectedSampleId;
            return (
              <li key={sample.sample_id}>
                <button
                  type="button"
                  onClick={() => onSelect(sample.sample_id)}
                  className={`flex w-full items-center justify-between gap-2 border-b border-slate-100 px-3 py-1.5 text-left transition-colors ${
                    active
                      ? "bg-slate-100 text-slate-900"
                      : "text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  <span className="font-mono text-[12px]">{shortId(sample.sample_id)}</span>
                  <span className="flex items-center gap-2 font-mono text-[10px] text-slate-500">
                    {sample.yolo_confidence !== null
                      ? sample.yolo_confidence.toFixed(2)
                      : "—"}
                    <span
                      className={
                        sample.decision === "Reliable"
                          ? "text-forest-700"
                          : sample.decision === "Unreliable"
                            ? "text-error-600"
                            : "text-warning-600"
                      }
                    >
                      {sample.decision ? sample.decision[0] : "·"}
                    </span>
                  </span>
                </button>
              </li>
            );
          })
        )}
      </ul>
    </section>
  );
}
