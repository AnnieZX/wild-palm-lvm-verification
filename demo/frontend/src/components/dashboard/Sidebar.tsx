import type { AblationCode, ModelInfo, VerificationDecision } from "@shared/types";

import { ConfidenceFilter } from "@/components/sidebar/ConfidenceFilter";
import { DecisionFilter } from "@/components/sidebar/DecisionFilter";
import { ModelSelector } from "@/components/sidebar/ModelSelector";
import { PromptSelector } from "@/components/sidebar/PromptSelector";
import { SampleSearch } from "@/components/sidebar/SampleSearch";

interface PromptOption {
  code: AblationCode;
  label: string;
  condition: string;
}

interface SidebarProps {
  models: ModelInfo[];
  selectedModelKey: string;
  onModelChange: (modelKey: string) => void;
  promptOptions: PromptOption[];
  selectedPrompt: string;
  onPromptChange: (code: string) => void;
  selectedDecision: string;
  onDecisionChange: (decision: string) => void;
  confidenceMin: number;
  confidenceMax: number;
  onConfidenceMinChange: (value: number) => void;
  onConfidenceMaxChange: (value: number) => void;
  searchSampleId: string;
  onSearchSampleIdChange: (value: string) => void;
  onSearchSubmit: () => void;
}

export function Sidebar({
  models,
  selectedModelKey,
  onModelChange,
  promptOptions,
  selectedPrompt,
  onPromptChange,
  selectedDecision,
  onDecisionChange,
  confidenceMin,
  confidenceMax,
  onConfidenceMinChange,
  onConfidenceMaxChange,
  searchSampleId,
  onSearchSampleIdChange,
  onSearchSubmit,
}: SidebarProps) {
  return (
    <aside
      className="flex h-full w-[280px] shrink-0 flex-col overflow-hidden border-r border-slate-200/80 bg-white"
      aria-label="Filters and navigation"
    >
      <div className="border-b border-slate-200/80 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-800">Controls</h2>
        <p className="text-xs text-slate-500">Model, prompt &amp; sample filters</p>
      </div>

      <nav className="flex-1 space-y-5 overflow-y-auto p-4">
        <ModelSelector
          models={models}
          selectedModelKey={selectedModelKey}
          onModelChange={onModelChange}
        />
        <PromptSelector
          options={promptOptions}
          selectedCode={selectedPrompt}
          onPromptChange={onPromptChange}
        />
        <ConfidenceFilter
          min={confidenceMin}
          max={confidenceMax}
          onMinChange={onConfidenceMinChange}
          onMaxChange={onConfidenceMaxChange}
        />
        <DecisionFilter
          selectedDecision={selectedDecision}
          onDecisionChange={onDecisionChange}
        />
        <SampleSearch
          value={searchSampleId}
          onChange={onSearchSampleIdChange}
          onSubmit={onSearchSubmit}
        />
      </nav>
    </aside>
  );
}
