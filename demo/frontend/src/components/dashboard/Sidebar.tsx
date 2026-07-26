import type { ModelInfo } from "@shared/types";

import { ConfidenceFilter } from "@/components/sidebar/ConfidenceFilter";
import { DecisionFilter } from "@/components/sidebar/DecisionFilter";
import { ModelSelector } from "@/components/sidebar/ModelSelector";
import { PromptSelector } from "@/components/sidebar/PromptSelector";
import { SampleSearch } from "@/components/sidebar/SampleSearch";
import { PROMPT_OPTIONS } from "@/lib/mock/dashboard";

interface SidebarProps {
  models: ModelInfo[];
  selectedModelKey: string;
  selectedPrompt: string;
  selectedDecision: string;
  confidenceMin: number;
  confidenceMax: number;
  searchSampleId: string;
}

export function Sidebar({
  models,
  selectedModelKey,
  selectedPrompt,
  selectedDecision,
  confidenceMin,
  confidenceMax,
  searchSampleId,
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
        <ModelSelector models={models} selectedModelKey={selectedModelKey} />
        <PromptSelector options={PROMPT_OPTIONS} selectedCode={selectedPrompt} />
        <ConfidenceFilter min={confidenceMin} max={confidenceMax} />
        <DecisionFilter selectedDecision={selectedDecision} />
        <SampleSearch value={searchSampleId} />
      </nav>
    </aside>
  );
}
