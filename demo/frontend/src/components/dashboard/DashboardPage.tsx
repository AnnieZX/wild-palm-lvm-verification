"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import type { SampleComparisonData } from "@/components/comparison/types";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Header } from "@/components/dashboard/Header";
import { InformationPanel } from "@/components/dashboard/InformationPanel";
import { MainViewer } from "@/components/dashboard/MainViewer";
import { Sidebar } from "@/components/dashboard/Sidebar";
import {
  fetchModels,
  fetchSampleDetail,
  fetchSamples,
  sampleImageUrl,
} from "@/lib/api";
import { DEFAULT_IMAGE_SIZE, PROMPT_OPTIONS } from "@/lib/constants";
import {
  buildSampleComparisonData,
  promptConditionForAblation,
  sampleResponseToComparisonEntry,
} from "@/lib/mappers";
import type {
  AblationCode,
  ModelInfo,
  SampleDetail,
  SampleSummary,
  VerificationDecision,
} from "@shared/types";

function pickDefaultModel(models: ModelInfo[]): ModelInfo | null {
  if (models.length === 0) {
    return null;
  }
  return models.find((model) => model.model_key === "qwen2_5_vl") ?? models[0];
}

/** Prefer multi-ablation A1 runs over newer single-ablation experiments. */
function pickDefaultExperiment(model: ModelInfo) {
  if (model.experiments.length === 0) {
    return null;
  }
  const withA1 = model.experiments.filter((experiment) =>
    experiment.ablations.includes("A1"),
  );
  const pool = withA1.length > 0 ? withA1 : model.experiments;
  return [...pool].sort((a, b) => {
    if (b.ablations.length !== a.ablations.length) {
      return b.ablations.length - a.ablations.length;
    }
    return b.sample_count - a.sample_count;
  })[0];
}

function filterByConfidence(
  samples: SampleSummary[],
  confidenceMin: number,
  confidenceMax: number,
): SampleSummary[] {
  return samples.filter((item) => {
    const confidence = item.yolo_confidence;
    if (confidence === null) {
      return true;
    }
    return confidence >= confidenceMin && confidence <= confidenceMax;
  });
}

export function DashboardPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedModelKey, setSelectedModelKey] = useState("");
  const [selectedExperimentId, setSelectedExperimentId] = useState("");
  const [selectedPrompt, setSelectedPrompt] = useState<AblationCode>("A1");
  const [selectedDecision, setSelectedDecision] = useState<VerificationDecision | "">("");
  const [confidenceMin, setConfidenceMin] = useState(0);
  const [confidenceMax, setConfidenceMax] = useState(1);
  const [searchSampleId, setSearchSampleId] = useState("");

  const [sampleList, setSampleList] = useState<SampleSummary[]>([]);
  const [sampleTotal, setSampleTotal] = useState(0);
  const [sample, setSample] = useState<SampleDetail | null>(null);
  const [comparison, setComparison] = useState<SampleComparisonData | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>(
    DEFAULT_IMAGE_SIZE,
  );
  const [detailLoading, setDetailLoading] = useState(false);

  const activeModel = useMemo(
    () => models.find((model) => model.model_key === selectedModelKey) ?? null,
    [models, selectedModelKey],
  );

  const activeExperiment = useMemo(
    () =>
      activeModel?.experiments.find(
        (experiment) => experiment.experiment_id === selectedExperimentId,
      ) ??
      activeModel?.experiments[0] ??
      null,
    [activeModel, selectedExperimentId],
  );

  const promptOptions = useMemo(() => {
    const available = new Set(activeExperiment?.ablations ?? []);
    return PROMPT_OPTIONS.filter((option) => available.has(option.code));
  }, [activeExperiment]);

  const scopeParams = useMemo(
    () => ({
      model_key: selectedModelKey,
      experiment_id: selectedExperimentId,
      ablation: selectedPrompt,
    }),
    [selectedModelKey, selectedExperimentId, selectedPrompt],
  );

  const loadSample = useCallback(
    async (sampleId: string) => {
      if (!selectedModelKey || !selectedExperimentId) {
        return;
      }

      setDetailLoading(true);
      try {
        const detailResponse = await fetchSampleDetail(sampleId, scopeParams);
        setSample(detailResponse.sample);
        setImageSrc(sampleImageUrl(sampleId, scopeParams));
        setError(null);

        const comparisonEntries = await Promise.all(
          models.map(async (model) => {
            const experiment = model.experiments[0];
            if (!experiment) {
              return null;
            }
            const ablation = experiment.ablations.includes(selectedPrompt)
              ? selectedPrompt
              : experiment.primary_ablation;

            try {
              const response = await fetchSampleDetail(sampleId, {
                model_key: model.model_key,
                experiment_id: experiment.experiment_id,
                ablation,
              });
              return sampleResponseToComparisonEntry(
                response,
                model,
                promptConditionForAblation(ablation),
              );
            } catch {
              return null;
            }
          }),
        );

        const predictions = comparisonEntries.filter(
          (entry): entry is NonNullable<typeof entry> => entry !== null,
        );

        setComparison(
          buildSampleComparisonData(
            detailResponse.sample,
            predictions.length > 0
              ? predictions
              : [
                  sampleResponseToComparisonEntry(
                    detailResponse,
                    activeModel ?? {
                      model_key: selectedModelKey,
                      display_name: selectedModelKey,
                      description: "",
                      experiments: [],
                    },
                    promptConditionForAblation(selectedPrompt),
                  ),
                ],
          ),
        );
      } catch (loadError) {
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load sample detail.",
        );
      } finally {
        setDetailLoading(false);
      }
    },
    [
      activeModel,
      models,
      scopeParams,
      selectedExperimentId,
      selectedModelKey,
      selectedPrompt,
    ],
  );

  const refreshSampleList = useCallback(async () => {
    if (!selectedModelKey || !selectedExperimentId) {
      return;
    }

    const pageSize = 200;
    const firstPage = await fetchSamples({
      ...scopeParams,
      page: 1,
      page_size: pageSize,
      decision: selectedDecision || undefined,
    });

    let allSamples = [...firstPage.samples];
    const totalPages = Math.max(1, Math.ceil(firstPage.total / pageSize));
    // Cap pages to keep the workstation responsive on large CSVs.
    const maxPages = Math.min(totalPages, 5);

    if (maxPages > 1) {
      const rest = await Promise.all(
        Array.from({ length: maxPages - 1 }, (_, index) =>
          fetchSamples({
            ...scopeParams,
            page: index + 2,
            page_size: pageSize,
            decision: selectedDecision || undefined,
          }),
        ),
      );
      for (const page of rest) {
        allSamples = allSamples.concat(page.samples);
      }
    }

    const candidates = filterByConfidence(
      allSamples,
      confidenceMin,
      confidenceMax,
    );
    setSampleList(candidates);
    setSampleTotal(candidates.length);
    return candidates;
  }, [
    confidenceMax,
    confidenceMin,
    scopeParams,
    selectedDecision,
    selectedExperimentId,
    selectedModelKey,
  ]);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetchModels();
        if (cancelled) {
          return;
        }

        const nextModels = response.models;
        setModels(nextModels);

        const defaultModel = pickDefaultModel(nextModels);
        if (!defaultModel) {
          setError("No experiment outputs found under outputs/verification/.");
          return;
        }

        const defaultExperiment = pickDefaultExperiment(defaultModel);
        if (!defaultExperiment) {
          setError("The selected model has no discovered experiments.");
          return;
        }

        setSelectedModelKey(defaultModel.model_key);
        setSelectedExperimentId(defaultExperiment.experiment_id);
        setSelectedPrompt(
          defaultExperiment.ablations.includes("A1")
            ? "A1"
            : defaultExperiment.primary_ablation,
        );
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Failed to load models from the API.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedModelKey || !selectedExperimentId || loading) {
      return;
    }

    let cancelled = false;

    async function refresh() {
      try {
        setError(null);
        const candidates = await refreshSampleList();
        if (cancelled || !candidates) {
          return;
        }

        const currentId = sample?.sample_id;
        const stillVisible =
          currentId !== undefined &&
          candidates.some((item) => item.sample_id === currentId);

        if (stillVisible && currentId) {
          return;
        }

        if (candidates[0]) {
          await loadSample(candidates[0].sample_id);
        } else {
          setSample(null);
          setComparison(null);
          setImageSrc(null);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Failed to load samples from the API.",
          );
        }
      }
    }

    void refresh();
    return () => {
      cancelled = true;
    };
    // Intentionally omit sample to avoid re-fetch loops when selecting within list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    loadSample,
    loading,
    refreshSampleList,
    selectedExperimentId,
    selectedModelKey,
    selectedPrompt,
    selectedDecision,
    confidenceMin,
    confidenceMax,
  ]);

  useEffect(() => {
    if (!imageSrc) {
      setImageSize(DEFAULT_IMAGE_SIZE);
      return;
    }

    const image = new Image();
    image.onload = () => {
      setImageSize({
        width: image.naturalWidth || DEFAULT_IMAGE_SIZE.width,
        height: image.naturalHeight || DEFAULT_IMAGE_SIZE.height,
      });
    };
    image.src = imageSrc;
  }, [imageSrc]);

  const handleModelChange = (modelKey: string) => {
    const model = models.find((item) => item.model_key === modelKey);
    const experiment = model ? pickDefaultExperiment(model) : null;
    setSelectedModelKey(modelKey);
    setSelectedExperimentId(experiment?.experiment_id ?? "");
    setSelectedPrompt(
      experiment?.ablations.includes("A1")
        ? "A1"
        : (experiment?.primary_ablation ?? "A1"),
    );
    setSearchSampleId("");
  };

  const selectedIndex = sampleList.findIndex(
    (item) => item.sample_id === sample?.sample_id,
  );
  const canPrevious = selectedIndex > 0;
  const canNext = selectedIndex >= 0 && selectedIndex < sampleList.length - 1;

  const goPrevious = () => {
    if (!canPrevious) return;
    const previous = sampleList[selectedIndex - 1];
    if (previous) void loadSample(previous.sample_id);
  };

  const goNext = () => {
    if (!canNext) return;
    const next = sampleList[selectedIndex + 1];
    if (next) void loadSample(next.sample_id);
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--wp-bg-app)] text-sm text-slate-600">
        Loading experiment catalog…
      </div>
    );
  }

  if (error && !sample && sampleList.length === 0) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--wp-bg-app)] px-6">
        <div className="max-w-lg rounded border border-red-200 bg-white p-5">
          <h1 className="text-base font-semibold text-slate-900">
            Workstation unavailable
          </h1>
          <p className="mt-2 text-sm text-slate-600">{error}</p>
          <p className="mt-3 text-xs text-slate-500">
            Ensure the FastAPI backend is running and can read repository{" "}
            <code className="rounded bg-slate-100 px-1">outputs/</code>.
          </p>
        </div>
      </div>
    );
  }

  if (!activeModel || !activeExperiment) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--wp-bg-app)] text-sm text-slate-600">
        No experiments available.
      </div>
    );
  }

  return (
    <DashboardShell
      header={
        <div className="relative">
          <Header
            experimentId={selectedExperimentId}
            sampleCount={activeExperiment.sample_count}
            modelName={activeModel.display_name}
            ablation={selectedPrompt}
          />
          <Link
            href="/statistics"
            className="absolute right-4 top-1/2 -translate-y-1/2 wp-link text-xs"
          >
            Statistics
          </Link>
        </div>
      }
      sidebar={
        <Sidebar
          models={models}
          selectedModelKey={selectedModelKey}
          onModelChange={handleModelChange}
          promptOptions={promptOptions}
          selectedPrompt={selectedPrompt}
          onPromptChange={(code) => setSelectedPrompt(code as AblationCode)}
          selectedDecision={selectedDecision}
          onDecisionChange={(decision) =>
            setSelectedDecision(decision as VerificationDecision | "")
          }
          confidenceMin={confidenceMin}
          confidenceMax={confidenceMax}
          onConfidenceMinChange={setConfidenceMin}
          onConfidenceMaxChange={setConfidenceMax}
          searchSampleId={searchSampleId}
          onSearchSampleIdChange={setSearchSampleId}
          onSearchSubmit={() => {
            const sampleId = searchSampleId.trim();
            if (sampleId) {
              void loadSample(sampleId);
              return;
            }
            if (sampleList[0]) {
              void loadSample(sampleList[0].sample_id);
            }
          }}
          samples={sampleList}
          selectedSampleId={sample?.sample_id ?? null}
          sampleTotal={sampleTotal}
          onSelectSample={(sampleId) => {
            void loadSample(sampleId);
          }}
          onPreviousSample={goPrevious}
          onNextSample={goNext}
          canPreviousSample={canPrevious}
          canNextSample={canNext}
        />
      }
      viewer={
        sample && comparison ? (
          <MainViewer
            sample={sample}
            activeModelKey={selectedModelKey}
            imageSrc={imageSrc}
            imageWidth={imageSize.width}
            imageHeight={imageSize.height}
            comparison={comparison}
          />
        ) : (
          <section className="flex min-h-0 flex-1 items-center justify-center bg-[var(--wp-bg-viewer)] text-sm text-slate-400">
            {detailLoading
              ? "Loading sample…"
              : error
                ? error
                : "No samples match the current filters."}
          </section>
        )
      }
      infoPanel={
        sample ? (
          <InformationPanel
            sample={sample}
            model={activeModel}
            experimentId={selectedExperimentId}
            ablation={selectedPrompt}
          />
        ) : (
          <aside className="flex h-full items-center justify-center border-l border-slate-200 bg-white text-xs text-slate-500">
            Select a sample to inspect.
          </aside>
        )
      }
    />
  );
}
