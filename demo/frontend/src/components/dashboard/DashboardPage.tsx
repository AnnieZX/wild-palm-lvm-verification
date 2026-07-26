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
  VerificationDecision,
} from "@shared/types";

function pickDefaultModel(models: ModelInfo[]): ModelInfo | null {
  if (models.length === 0) {
    return null;
  }
  return (
    models.find((model) => model.model_key === "qwen2_5_vl") ??
    models[0]
  );
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

  const [sample, setSample] = useState<SampleDetail | null>(null);
  const [comparison, setComparison] = useState<SampleComparisonData | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>(
    DEFAULT_IMAGE_SIZE,
  );

  const activeModel = useMemo(
    () => models.find((model) => model.model_key === selectedModelKey) ?? null,
    [models, selectedModelKey],
  );

  const activeExperiment = useMemo(
    () =>
      activeModel?.experiments.find(
        (experiment) => experiment.experiment_id === selectedExperimentId,
      ) ?? activeModel?.experiments[0] ?? null,
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

      const detailResponse = await fetchSampleDetail(sampleId, scopeParams);
      setSample(detailResponse.sample);
      setImageSrc(sampleImageUrl(sampleId, scopeParams));

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

  const loadFirstMatchingSample = useCallback(async () => {
    if (!selectedModelKey || !selectedExperimentId) {
      return;
    }

    const listResponse = await fetchSamples({
      ...scopeParams,
      page: 1,
      page_size: 50,
      decision: selectedDecision || undefined,
    });

    let candidates = listResponse.samples;
    candidates = candidates.filter((item) => {
      const confidence = item.yolo_confidence;
      if (confidence === null) {
        return true;
      }
      return confidence >= confidenceMin && confidence <= confidenceMax;
    });

    const targetId = candidates[0]?.sample_id;
    if (!targetId) {
      setSample(null);
      setComparison(null);
      setImageSrc(null);
      return;
    }

    await loadSample(targetId);
  }, [
    confidenceMax,
    confidenceMin,
    loadSample,
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

        const defaultExperiment = defaultModel.experiments[0];
        if (!defaultExperiment) {
          setError("The selected model has no discovered experiments.");
          return;
        }

        setSelectedModelKey(defaultModel.model_key);
        setSelectedExperimentId(defaultExperiment.experiment_id);
        setSelectedPrompt(defaultExperiment.primary_ablation);
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

    async function refreshSamples() {
      try {
        setError(null);
        await loadFirstMatchingSample();
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

    void refreshSamples();
    return () => {
      cancelled = true;
    };
  }, [loadFirstMatchingSample, loading, selectedExperimentId, selectedModelKey]);

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
    const experiment = model?.experiments[0];
    setSelectedModelKey(modelKey);
    setSelectedExperimentId(experiment?.experiment_id ?? "");
    setSelectedPrompt(experiment?.primary_ablation ?? "A1");
    setSearchSampleId("");
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-100 text-sm text-slate-600">
        Loading experiment catalog…
      </div>
    );
  }

  if (error && !sample) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-100 px-6">
        <div className="max-w-lg rounded-xl border border-red-200 bg-white p-6 shadow-sm">
          <h1 className="text-lg font-semibold text-slate-900">Dashboard unavailable</h1>
          <p className="mt-2 text-sm text-slate-600">{error}</p>
          <p className="mt-4 text-xs text-slate-500">
            Ensure the FastAPI backend is running and{" "}
            <code className="rounded bg-slate-100 px-1">DEMO_OUTPUTS_ROOT</code> points at
            your experiment outputs.
          </p>
        </div>
      </div>
    );
  }

  if (!activeModel || !activeExperiment || !sample || !comparison) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-100 text-sm text-slate-600">
        No samples available for the selected experiment.
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
            void loadFirstMatchingSample();
          }}
        />
      }
      viewer={
        <MainViewer
          sample={sample}
          activeModelKey={selectedModelKey}
          imageSrc={imageSrc}
          imageWidth={imageSize.width}
          imageHeight={imageSize.height}
          comparison={comparison}
        />
      }
      infoPanel={
        <InformationPanel
          sample={sample}
          model={activeModel}
          experimentId={selectedExperimentId}
          ablation={selectedPrompt}
        />
      }
    />
  );
}
