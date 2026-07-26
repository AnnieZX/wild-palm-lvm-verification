interface ConfidenceFilterProps {
  min: number;
  max: number;
}

export function ConfidenceFilter({ min, max }: ConfidenceFilterProps) {
  return (
    <fieldset className="space-y-3">
      <legend className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Confidence filter
      </legend>
      <div className="grid grid-cols-2 gap-2">
        <label className="block text-[11px] text-slate-500">
          Min
          <input
            type="number"
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 font-mono text-sm"
            defaultValue={min}
            min={0}
            max={1}
            step={0.01}
            disabled
            aria-label="Minimum confidence (placeholder)"
          />
        </label>
        <label className="block text-[11px] text-slate-500">
          Max
          <input
            type="number"
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 font-mono text-sm"
            defaultValue={max}
            min={0}
            max={1}
            step={0.01}
            disabled
            aria-label="Maximum confidence (placeholder)"
          />
        </label>
      </div>
      <input
        type="range"
        className="w-full accent-forest-700"
        min={0}
        max={100}
        defaultValue={Math.round(((min + max) / 2) * 100)}
        disabled
        aria-label="Confidence range (placeholder)"
      />
    </fieldset>
  );
}
