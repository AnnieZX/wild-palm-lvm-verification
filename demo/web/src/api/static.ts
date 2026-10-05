import type {
  Condition,
  ContextShift,
  CuratedCase,
  DetectionDetail,
  DetectionList,
  DetectionSummary,
  Difficulty,
  ModelInfo,
  Predictions,
  PromptInfo,
  Reasoning,
  ReviewDecks,
  ReviewResultItem,
  ReviewResults,
  ReviewSession,
  Scaling,
  SemanticReview,
  Summary,
  VisitorLabel,
} from "../types/api";
import { ApiError, type DetectionFilters } from "./client";

/**
 * Read-only client for the exported snapshot (demo/backend/scripts/export_static.py).
 * Mirrors the live API's response shapes; filtering, paging and review sessions run in the browser.
 */

export interface SnapshotMeta {
  generated: string;
  n_detections_total: number;
  n_detections_included: number;
  review_pool: Record<string, number>;
  image_encoding: string;
}

type IndexRow = DetectionSummary & { a1_a5_changed_models: string[] };
interface DetectionBundle {
  detail: DetectionDetail;
  predictions: Predictions;
  prompts: Record<Condition, PromptInfo | null>;
}
interface PoolItem {
  key: string;
  image_url: string;
  bbox_xywh: [number, number, number, number];
  image_width: number | null;
  image_height: number | null;
}
type RevealEntry = Omit<ReviewResultItem, "index" | "image_url" | "bbox_xywh" | "image_width" | "image_height" | "visitor_label" | "agrees">;

const BASE = import.meta.env.BASE_URL;
const cache = new Map<string, Promise<unknown>>();

function asset(url: string): string {
  return url.startsWith("snapshot/") ? `${BASE}${url}` : url;
}

function load<T>(rel: string): Promise<T> {
  let p = cache.get(rel) as Promise<T> | undefined;
  if (!p) {
    p = fetch(`${BASE}snapshot/${rel}`).then((res) => {
      if (!res.ok) throw new ApiError(res.status, res.status === 404 ? "not included in this snapshot" : res.statusText);
      return res.json() as Promise<T>;
    });
    p.catch(() => cache.delete(rel));
    cache.set(rel, p);
  }
  return p;
}

const pairFile = (pair: string) => pair.replace("->", "-");

function matches(d: IndexRow, f: DetectionFilters): boolean {
  if (f.q) {
    let needle = f.q.trim().toLowerCase();
    if (/^\d+$/.test(needle)) needle = `sample_${needle.padStart(6, "0")}`;
    if (!d.detection_id.toLowerCase().includes(needle) && !d.image_id.toLowerCase().includes(needle)) return false;
  }
  if (f.matched !== undefined && f.matched !== null && d.labelme_matched !== f.matched) return false;
  if (f.semantic) {
    if (f.semantic === "not_reviewed" ? d.semantic_label !== null : d.semantic_label !== f.semantic) return false;
  }
  if (f.conf_q && d.confidence_quintile !== f.conf_q) return false;
  if (f.area_q && d.area_quintile !== f.area_q) return false;
  if (f.changed_model && !d.a1_a5_changed_models.includes(f.changed_model)) return false;
  return true;
}

async function bundle(id: string): Promise<DetectionBundle> {
  const b = await load<DetectionBundle>(`detections/${id}.json`);
  return b;
}

interface Session {
  id: string;
  deck: string;
  items: PoolItem[];
  labels: Map<number, VisitorLabel>;
}
const sessions = new Map<string, Session>();

function randomId(): string {
  const a = new Uint8Array(12);
  crypto.getRandomValues(a);
  return Array.from(a, (x) => x.toString(16).padStart(2, "0")).join("");
}

function sample<T>(xs: T[], n: number): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

async function publicSession(s: Session): Promise<ReviewSession> {
  const decks = await load<ReviewDecks>("review/decks.json");
  return {
    session_id: s.id,
    deck: s.deck,
    deck_info: decks.decks[s.deck],
    question: decks.question,
    labels: decks.labels,
    definitions: decks.definitions,
    items: s.items.map((it, index) => ({
      index,
      image_url: asset(it.image_url),
      bbox_xywh: it.bbox_xywh,
      image_width: it.image_width,
      image_height: it.image_height,
      visitor_label: s.labels.get(index) ?? null,
    })),
    n_answered: s.labels.size,
    complete: s.labels.size === s.items.length,
  };
}

function remember(event: Record<string, unknown>): void {
  try {
    const key = "wildpalm.visitor_reviews";
    const log = JSON.parse(localStorage.getItem(key) ?? "[]") as unknown[];
    log.push({ ...event, time: new Date().toISOString() });
    localStorage.setItem(key, JSON.stringify(log.slice(-2000)));
  } catch {
    /* storage unavailable: answers stay in memory only */
  }
}

export const snapshotMeta = () => load<SnapshotMeta>("meta.json");

export const staticApi = {
  getSummary: () => load<Summary>("summary.json"),
  getModels: () => load<{ models: ModelInfo[] }>("models.json").then((r) => r.models),

  listDetections: async (filters: DetectionFilters, page = 1, pageSize = 50): Promise<DetectionList> => {
    const rows = (await load<IndexRow[]>("detections/index.json")).filter((d) => matches(d, filters));
    const start = Math.max(0, (page - 1) * pageSize);
    return { total: rows.length, page, page_size: pageSize, items: rows.slice(start, start + pageSize) };
  },
  randomDetection: async (filters: DetectionFilters) => {
    const rows = (await load<IndexRow[]>("detections/index.json")).filter((d) => matches(d, filters));
    if (!rows.length) throw new ApiError(404, "no detection matches the filters");
    return rows[Math.floor(Math.random() * rows.length)].detection_id;
  },
  getCurated: () => load<{ available: boolean; cases: CuratedCase[] }>("curated.json").then((r) => r.cases),
  getDetection: async (id: string): Promise<DetectionDetail> => {
    const d = (await bundle(id)).detail;
    const inputs = Object.fromEntries(
      Object.entries(d.condition_inputs).map(([c, v]) => [c, { ...v, image_url: asset(v.image_url ?? "") }]),
    ) as DetectionDetail["condition_inputs"];
    return { ...d, raw_image_url: asset(d.raw_image_url), condition_inputs: inputs };
  },
  getPredictions: async (id: string) => (await bundle(id)).predictions,
  getReasoning: async (id: string, model: string, condition: Condition): Promise<Reasoning> => {
    const all = await load<Record<string, Partial<Record<Condition, Reasoning>>>>(`reasoning/${id}.json`);
    return all[model]?.[condition] ?? ({ available: false, reason: "no usable prediction for this cell" } as Reasoning);
  },
  getPrompt: async (id: string, condition: Condition): Promise<PromptInfo> => {
    const p = (await bundle(id)).prompts[condition];
    if (!p) throw new ApiError(404, "prompt not found");
    return p;
  },

  getContextShift: (pair: string) => load<ContextShift>(`analysis/context-shift/${pairFile(pair)}.json`),
  getScaling: () => load<Scaling>("analysis/scaling.json"),
  getDifficulty: (pair: string) => load<Difficulty>(`analysis/difficulty/${pairFile(pair)}.json`),
  getSemanticReview: () => load<SemanticReview>("analysis/semantic-review.json"),

  getReviewDecks: () => load<ReviewDecks>("review/decks.json"),
  createReviewSession: async (count: number, deck: string): Promise<ReviewSession> => {
    const pool = await load<PoolItem[]>(`review/${deck}/pool.json`);
    const s: Session = { id: randomId(), deck, items: sample(pool, Math.min(count, pool.length)), labels: new Map() };
    sessions.set(s.id, s);
    return publicSession(s);
  },
  submitReview: async (sessionId: string, index: number, label: VisitorLabel): Promise<ReviewSession> => {
    const s = sessions.get(sessionId);
    if (!s) throw new ApiError(404, "unknown session");
    if (index < 0 || index >= s.items.length) throw new ApiError(422, "item index out of range");
    s.labels.set(index, label);
    return publicSession(s);
  },
  getReviewResults: async (sessionId: string): Promise<ReviewResults> => {
    const s = sessions.get(sessionId);
    if (!s) throw new ApiError(404, "unknown session");
    if (s.labels.size < s.items.length) {
      throw new ApiError(403, "reveal is available only after every item is answered or skipped");
    }
    const [reveal, decks, models] = await Promise.all([
      load<Record<string, RevealEntry>>(`review/${s.deck}/reveal.json`),
      load<ReviewDecks>("review/decks.json"),
      load<ReviewResults["models"]>("review/models.json"),
    ]);
    const items: ReviewResultItem[] = s.items.map((it, index) => {
      const visitor = s.labels.get(index)!;
      const r = reveal[it.key];
      return {
        ...r,
        index,
        image_url: asset(it.image_url),
        bbox_xywh: it.bbox_xywh,
        image_width: it.image_width,
        image_height: it.image_height,
        visitor_label: visitor,
        agrees: visitor === "skip" ? null : visitor === r.research_label,
      };
    });
    const answered = items.filter((i) => i.visitor_label !== "skip");
    remember({ deck: s.deck, labels: items.map((i) => ({ ref_id: i.ref_id, label: i.visitor_label })) });
    return {
      session_id: s.id,
      deck: s.deck,
      deck_info: decks.decks[s.deck],
      items,
      n_answered: answered.length,
      n_agree: answered.filter((i) => i.agrees).length,
      models,
    };
  },
};
