interface SampleSearchProps {
  value: string;
}

export function SampleSearch({ value }: SampleSearchProps) {
  return (
    <fieldset className="space-y-2">
      <legend className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Search sample ID
      </legend>
      <input
        type="search"
        className="w-full rounded-md border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-sm placeholder:text-slate-400"
        placeholder="sample_000042"
        defaultValue={value}
        disabled
        aria-label="Search sample ID (placeholder)"
      />
    </fieldset>
  );
}
