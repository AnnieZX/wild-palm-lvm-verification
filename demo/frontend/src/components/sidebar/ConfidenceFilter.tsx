interface ConfidenceFilterProps {
  min: number;
  max: number;
  onMinChange: (value: number) => void;
  onMaxChange: (value: number) => void;
}

export function ConfidenceFilter({
  min,
  max,
  onMinChange,
  onMaxChange,
}: ConfidenceFilterProps) {
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
            value={min}
            min={0}
            max={1}
            step={0.01}
            onChange={(event) => onMinChange(Number(event.target.value))}
            aria-label="Minimum confidence"
          />
        </label>
        <label className="block text-[11px] text-slate-500">
          Max
          <input
            type="number"
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 font-mono text-sm"
            value={max}
            min={0}
            max={1}
            step={0.01}
            onChange={(event) => onMaxChange(Number(event.target.value))}
            aria-label="Maximum confidence"
          />
        </label>
      </div>
      <input
        type="range"
        className="w-full accent-forest-700"
        min={0}
        max={100}
        value={Math.round(((min + max) / 2) * 100)}
        readOnly
        aria-label="Confidence range midpoint"
      />
    </fieldset>
  );
}
