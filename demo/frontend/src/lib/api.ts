/**
 * Browser HTTP client for the demo FastAPI backend.
 * Frontend never reads outputs/ directly.
 */

import type {
  ExperimentQueryParams,
  ModelsResponse,
  SampleDetailResponse,
  SampleListQueryParams,
  SampleListResponse,
  StatisticsResponse,
} from "@shared/types";

const DEFAULT_API_BASE = "http://localhost:8000";

export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  return (configured && configured.length > 0 ? configured : DEFAULT_API_BASE).replace(
    /\/$/,
    "",
  );
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function buildQuery(
  params: Record<string, string | number | boolean | undefined | null>,
): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `?${query}` : "";
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // keep default message
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export async function fetchHealth(): Promise<{ status: string; service: string }> {
  return requestJson("/api/v1/health");
}

export async function fetchModels(): Promise<ModelsResponse> {
  return requestJson("/api/v1/models");
}

export async function fetchStatistics(
  params: ExperimentQueryParams = {},
): Promise<StatisticsResponse> {
  return requestJson(
    `/api/v1/statistics${buildQuery({
      model_key: params.model_key,
      experiment_id: params.experiment_id,
      ablation: params.ablation,
    })}`,
  );
}

export async function fetchSamples(
  params: SampleListQueryParams = {},
): Promise<SampleListResponse> {
  return requestJson(
    `/api/v1/samples${buildQuery({
      model_key: params.model_key,
      experiment_id: params.experiment_id,
      ablation: params.ablation,
      page: params.page,
      page_size: params.page_size,
      decision: params.decision,
      matched_gt: params.matched_gt,
      gt_label: params.gt_label,
    })}`,
  );
}

export async function fetchSampleDetail(
  sampleId: string,
  params: ExperimentQueryParams = {},
): Promise<SampleDetailResponse> {
  return requestJson(
    `/api/v1/sample/${encodeURIComponent(sampleId)}${buildQuery({
      model_key: params.model_key,
      experiment_id: params.experiment_id,
      ablation: params.ablation,
    })}`,
  );
}

/** Absolute URL for an ablation overlay image (browser <img src>). */
export function sampleImageUrl(
  sampleId: string,
  params: ExperimentQueryParams = {},
): string {
  return `${getApiBaseUrl()}/api/v1/image/${encodeURIComponent(sampleId)}${buildQuery({
    model_key: params.model_key,
    experiment_id: params.experiment_id,
    ablation: params.ablation,
  })}`;
}
