"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { StatisticsDashboard } from "@/components/statistics";
import type { StatisticsDashboardData } from "@/components/statistics/types";
import { fetchModels, fetchStatistics } from "@/lib/api";
import { statisticsToDashboardData } from "@/lib/mappers";
import type { AblationCode, ModelInfo, StatisticsResponse } from "@shared/types";

function pickDefaultModel(models: ModelInfo[]): ModelInfo | null {
  if (models.length === 0) {
    return null;
  }
  return models.find((model) => model.model_key === "qwen2_5_vl") ?? models[0];
}

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

async function fetchStatisticsSafe(
  modelKey: string,
  experimentId: string,
  ablation: AblationCode,
): Promise<StatisticsResponse | null> {
  try {
    return await fetchStatistics({
      model_key: modelKey,
      experiment_id: experimentId,
      ablation,
    });
  } catch {
    return null;
  }
}

export function StatisticsPageClient() {
  const [data, setData] = useState<StatisticsDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStatistics() {
      try {
        setLoading(true);
        setError(null);

        const modelsResponse = await fetchModels();
        const models = modelsResponse.models;
        const primaryModel = pickDefaultModel(models);
        const primaryExperiment = primaryModel
          ? pickDefaultExperiment(primaryModel)
          : null;

        if (!primaryModel || !primaryExperiment) {
          throw new Error("No experiment statistics available.");
        }

        const primaryAblation = primaryExperiment.ablations.includes("A1")
          ? ("A1" as AblationCode)
          : primaryExperiment.primary_ablation;
        const primaryStats = await fetchStatistics({
          model_key: primaryModel.model_key,
          experiment_id: primaryExperiment.experiment_id,
          ablation: primaryAblation,
        });

        const modelStatistics = (
          await Promise.all(
            models.map(async (model) => {
              const experiment = model.experiments[0];
              if (!experiment) {
                return null;
              }
              return fetchStatisticsSafe(
                model.model_key,
                experiment.experiment_id,
                experiment.primary_ablation,
              );
            }),
          )
        ).filter((entry): entry is StatisticsResponse => entry !== null);

        const ablationStatistics = (
          await Promise.all(
            primaryExperiment.ablations.map((ablation) =>
              fetchStatisticsSafe(
                primaryModel.model_key,
                primaryExperiment.experiment_id,
                ablation,
              ),
            ),
          )
        ).filter((entry): entry is StatisticsResponse => entry !== null);

        if (cancelled) {
          return;
        }

        setData(
          statisticsToDashboardData(
            primaryStats,
            modelStatistics,
            ablationStatistics,
            models,
          ),
        );
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Failed to load statistics from the API.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadStatistics();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen overflow-y-auto bg-[var(--wp-bg-app)]">
      <nav className="border-b border-slate-200 bg-white px-4 py-2">
        <div className="mx-auto flex max-w-7xl items-center gap-3">
          <Link href="/" className="wp-link text-xs">
            ← Inspection
          </Link>
          <span className="text-slate-300">|</span>
          <span className="text-sm font-medium text-slate-700">Statistics</span>
        </div>
      </nav>

      <main className="mx-auto max-w-7xl px-4 py-5">
        {loading ? (
          <p className="text-sm text-slate-600">Loading statistics…</p>
        ) : error ? (
          <div className="rounded border border-red-200 bg-white p-5">
            <h1 className="text-base font-semibold text-slate-900">
              Statistics unavailable
            </h1>
            <p className="mt-2 text-sm text-slate-600">{error}</p>
          </div>
        ) : data ? (
          <StatisticsDashboard data={data} />
        ) : null}
      </main>
    </div>
  );
}
