import type {
  Condition,
  ContextShift,
  CuratedCase,
  DetectionDetail,
  DetectionList,
  Difficulty,
  ModelInfo,
  Predictions,
  PromptInfo,
  Quintile,
  Reasoning,
  ReviewDecks,
  ReviewResults,
  ReviewSession,
  Scaling,
  SemanticReview,
  Summary,
  VisitorLabel,
} from "../types/api";
import { staticApi } from "./static";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : "";
}

export interface DetectionFilters {
  q?: string;
  matched?: boolean | null;
  semantic?: "palm" | "ambiguous" | "non_palm" | "not_reviewed" | null;
  conf_q?: Quintile | null;
  area_q?: Quintile | null;
  changed_model?: string | null;
}

export const IS_STATIC = import.meta.env.VITE_STATIC === "1";

const liveApi = {
  getSummary: () => request<Summary>("/api/summary"),
  getModels: () => request<{ models: ModelInfo[] }>("/api/models").then((r) => r.models),

  listDetections: (filters: DetectionFilters, page = 1, pageSize = 50) =>
    request<DetectionList>(`/api/detections${qs({ ...filters, page, page_size: pageSize })}`),
  randomDetection: (filters: DetectionFilters) =>
    request<{ detection_id: string }>(`/api/detections/random${qs({ ...filters })}`).then((r) => r.detection_id),
  getCurated: () =>
    request<{ available: boolean; cases: CuratedCase[] }>("/api/detections/curated").then((r) => r.cases),
  getDetection: (id: string) => request<DetectionDetail>(`/api/detections/${encodeURIComponent(id)}`),
  getPredictions: (id: string) => request<Predictions>(`/api/detections/${encodeURIComponent(id)}/predictions`),
  getReasoning: (id: string, model: string, condition: Condition) =>
    request<Reasoning>(`/api/detections/${encodeURIComponent(id)}/reasoning${qs({ model, condition })}`),
  getPrompt: (id: string, condition: Condition) =>
    request<PromptInfo>(`/api/detections/${encodeURIComponent(id)}/prompt/${condition}`),

  getContextShift: (pair: string) => request<ContextShift>(`/api/analysis/context-shift${qs({ pair })}`),
  getScaling: () => request<Scaling>("/api/analysis/scaling"),
  getDifficulty: (pair: string) => request<Difficulty>(`/api/analysis/difficulty${qs({ pair })}`),
  getSemanticReview: () => request<SemanticReview>("/api/analysis/semantic-review"),

  getReviewDecks: () => request<ReviewDecks>("/api/review/decks"),
  createReviewSession: (count: number, deck: string) =>
    request<ReviewSession>("/api/review/sessions", { method: "POST", body: JSON.stringify({ count, deck }) }),
  submitReview: (sessionId: string, index: number, label: VisitorLabel) =>
    request<ReviewSession>(`/api/review/sessions/${sessionId}/labels`, {
      method: "POST",
      body: JSON.stringify({ index, label }),
    }),
  getReviewResults: (sessionId: string) => request<ReviewResults>(`/api/review/sessions/${sessionId}/results`),
};

export const api: typeof liveApi = IS_STATIC ? staticApi : liveApi;
