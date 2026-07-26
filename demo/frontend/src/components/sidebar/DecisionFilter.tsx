import { DECISION_OPTIONS } from "@/lib/constants";

interface DecisionFilterProps {
  selectedDecision: string;
  onDecisionChange: (decision: string) => void;
}

export function DecisionFilter({
  selectedDecision,
  onDecisionChange,
}: DecisionFilterProps) {
  return (
    <fieldset className="space-y-2">
      <legend className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Decision filter
      </legend>
      <div className="flex flex-wrap gap-2">
        <label className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs">
          <input
            type="radio"
            name="decision-filter"
            value=""
            checked={selectedDecision === ""}
            onChange={() => onDecisionChange("")}
          />
          All
        </label>
        {DECISION_OPTIONS.map((decision) => (
          <label
            key={decision}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs"
          >
            <input
              type="radio"
              name="decision-filter"
              value={decision}
              checked={selectedDecision === decision}
              onChange={() => onDecisionChange(decision)}
            />
            {decision}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
