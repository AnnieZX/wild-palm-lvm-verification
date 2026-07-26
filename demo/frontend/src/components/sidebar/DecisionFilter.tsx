import { DECISION_OPTIONS } from "@/lib/mock/dashboard";

interface DecisionFilterProps {
  selectedDecision: string;
}

export function DecisionFilter({ selectedDecision }: DecisionFilterProps) {
  return (
    <fieldset className="space-y-2">
      <legend className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Decision filter
      </legend>
      <div className="flex flex-wrap gap-2">
        <label className="inline-flex cursor-not-allowed items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs">
          <input
            type="radio"
            name="decision-filter"
            value=""
            defaultChecked={selectedDecision === ""}
            disabled
          />
          All
        </label>
        {DECISION_OPTIONS.map((decision) => (
          <label
            key={decision}
            className="inline-flex cursor-not-allowed items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs"
          >
            <input
              type="radio"
              name="decision-filter"
              value={decision}
              defaultChecked={selectedDecision === decision}
              disabled
            />
            {decision}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
