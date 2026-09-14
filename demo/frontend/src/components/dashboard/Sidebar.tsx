import type { AblationCode, ModelInfo, SampleSummary } from "@shared/types";

import { ConfidenceFilter } from "@/components/sidebar/ConfidenceFilter";
import { DecisionFilter } from "@/components/sidebar/DecisionFilter";
import { ModelSelector } from "@/components/sidebar/ModelSelector";
import { PromptSelector } from "@/components/sidebar/PromptSelector";
import { SampleNavigator } from "@/components/sidebar/SampleNavigator";
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
  samples: SampleSummary[];
  selectedSampleId: string | null;
  sampleTotal: number;
  onSelectSample: (sampleId: string) => void;
  onPreviousSample: () => void;
  onNextSample: () => void;
  canPreviousSample: boolean;
  canNextSample: boolean;
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
  samples,
  selectedSampleId,
  sampleTotal,
  onSelectSample,
  onPreviousSample,
  onNextSample,
  canPreviousSample,
  canNextSample,
}: SidebarProps) {
  return (
    <aside
      className="flex h-full w-[260px] shrink-0 flex-col overflow-hidden border-r border-slate-200 bg-white"
      aria-label="Filters and results"
    >
      <div className="border-b border-slate-200 px-3 py-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Filters
        </h2>
      </div>

      <nav className="shrink-0 space-y-4 overflow-y-auto border-b border-slate-200 p-3">
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
        <DecisionFilter
          selectedDecision={selectedDecision}
          onDecisionChange={onDecisionChange}
        />
        <ConfidenceFilter
          min={confidenceMin}
          max={confidenceMax}
          onMinChange={onConfidenceMinChange}
          onMaxChange={onConfidenceMaxChange}
        />
        <SampleSearch
          value={searchSampleId}
          onChange={onSearchSampleIdChange}
          onSubmit={onSearchSubmit}
        />
      </nav>

      <SampleNavigator
        samples={samples}
        selectedSampleId={selectedSampleId}
        total={sampleTotal}
        onSelect={onSelectSample}
        onPrevious={onPreviousSample}
        onNext={onNextSample}
        canPrevious={canPreviousSample}
        canNext={canNextSample}
      />
    </aside>
  );
}
