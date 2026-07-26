import type { ModelInfo } from "@shared/types";

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModelKey: string;
}

export function ModelSelector({ models, selectedModelKey }: ModelSelectorProps) {
  return (
    <fieldset className="space-y-2">
      <legend className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Model
      </legend>
      <select
        className="w-full rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 shadow-sm"
        defaultValue={selectedModelKey}
        disabled
        aria-label="Model selector (placeholder)"
      >
        {models.map((model) => (
          <option key={model.model_key} value={model.model_key}>
            {model.display_name}
          </option>
        ))}
      </select>
    </fieldset>
  );
}
