import Link from "next/link";

import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Header } from "@/components/dashboard/Header";
import { InformationPanel } from "@/components/dashboard/InformationPanel";
import { MainViewer } from "@/components/dashboard/MainViewer";
import { Sidebar } from "@/components/dashboard/Sidebar";
import { MOCK_DASHBOARD_STATE, MOCK_MODELS } from "@/lib/mock/dashboard";

export default function HomePage() {
  const state = MOCK_DASHBOARD_STATE;
  const activeModel =
    MOCK_MODELS.find((model) => model.model_key === state.selectedModelKey) ?? MOCK_MODELS[0];
  const experiment = activeModel.experiments[0];

  return (
    <DashboardShell
      header={
        <div className="relative">
          <Header
            experimentId={state.selectedExperimentId}
            sampleCount={experiment?.sample_count ?? 0}
          />
          <Link
            href="/statistics"
            className="absolute right-5 top-1/2 -translate-y-1/2 wp-link"
          >
            Statistics →
          </Link>
        </div>
      }
      sidebar={
        <Sidebar
          models={MOCK_MODELS}
          selectedModelKey={state.selectedModelKey}
          selectedPrompt={state.selectedPrompt}
          selectedDecision={state.selectedDecision}
          confidenceMin={state.confidenceRange.min}
          confidenceMax={state.confidenceRange.max}
          searchSampleId={state.searchSampleId}
        />
      }
      viewer={
        <MainViewer sample={state.sample} activeModelKey={state.selectedModelKey} />
      }
      infoPanel={
        <InformationPanel
          sample={state.sample}
          model={activeModel}
          experimentId={state.selectedExperimentId}
          ablation={state.selectedPrompt}
        />
      }
    />
  );
}
