import type { AblationCode } from "@shared/types";

interface PromptOption {
  code: AblationCode;
  label: string;
  condition: string;
}

interface PromptSelectorProps {
  options: PromptOption[];
  selectedCode: string;
  onPromptChange: (code: string) => void;
}

export function PromptSelector({
  options,
  selectedCode,
  onPromptChange,
}: PromptSelectorProps) {
  return (
    <fieldset className="space-y-2">
      <legend className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Prompt / Ablation
      </legend>
      <select
        className="w-full rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 shadow-sm"
        value={selectedCode}
        onChange={(event) => onPromptChange(event.target.value)}
        aria-label="Prompt selector"
      >
        {options.map((option) => (
          <option key={option.code} value={option.code}>
            {option.label}
          </option>
        ))}
      </select>
      <p className="font-mono text-[10px] text-slate-400">
        {options.find((option) => option.code === selectedCode)?.condition}
      </p>
    </fieldset>
  );
}
