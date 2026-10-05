import { useEffect, useMemo, useState } from "react";
import { api, type DetectionFilters } from "../../api/client";
import { useApi } from "../../hooks/useApi";
import type { CuratedCase, DetectionList, ModelInfo, Quintile } from "../../types/api";
import { humanize, num } from "../../utils/format";
import { SemanticBadge } from "../common/Labels";
import { Segmented } from "../common/Segmented";

const QUINTILES: Quintile[] = ["Q1", "Q2", "Q3", "Q4", "Q5"];

interface Props {
  selected: string | null;
  onSelect: (id: string) => void;
  models: ModelInfo[];
}

export function DetectionBrowser({ selected, onSelect, models }: Props) {
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [matched, setMatched] = useState<"all" | "matched" | "unmatched">("all");
  const [semantic, setSemantic] = useState<"" | "palm" | "ambiguous" | "not_reviewed">("");
  const [confQ, setConfQ] = useState<Quintile | null>(null);
  const [areaQ, setAreaQ] = useState<Quintile | null>(null);
  const [changed, setChanged] = useState("");
  const [page, setPage] = useState(1);
  const [curatedCat, setCuratedCat] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setQuery(q), 250);
    return () => clearTimeout(t);
  }, [q]);

  const filters: DetectionFilters = useMemo(
    () => ({
      q: query || undefined,
      matched: matched === "all" ? null : matched === "matched",
      semantic: semantic || null,
      conf_q: confQ,
      area_q: areaQ,
      changed_model: changed || null,
    }),
    [query, matched, semantic, confQ, areaQ, changed],
  );
  useEffect(() => setPage(1), [filters]);

  const list = useApi<DetectionList>(() => api.listDetections(filters, page, 60), [filters, page]);
  const curated = useApi<CuratedCase[]>(() => api.getCurated(), []);
  const categories = useMemo(
    () => Array.from(new Set((curated.data ?? []).map((c) => c.category))),
    [curated.data],
  );
  const changeModels = models.filter((m) => m.conditions.A1.status !== "not_run" && m.conditions.A5.n_valid);

  const random = async () => {
    try {
      onSelect(await api.randomDetection(filters));
    } catch {
      /* no match: list already shows zero results */
    }
  };

  const pages = list.data ? Math.max(1, Math.ceil(list.data.total / list.data.page_size)) : 1;

  return (
    <aside className="browser">
      <div className="row">
        <input
          className="input"
          placeholder="Detection ID, number or patch name"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ flex: 1 }}
        />
        <button className="btn" onClick={random} title="Random detection matching the filters">
          Random
        </button>
      </div>

      <div>
        <span className="field-label">LabelMe geometry (IoU ≥ 0.5)</span>
        <Segmented
          value={matched}
          onChange={setMatched}
          options={[
            { value: "all", label: "All" },
            { value: "matched", label: "Matched" },
            { value: "unmatched", label: "Unmatched" },
          ]}
        />
      </div>
      <div>
        <span className="field-label">Human semantic review</span>
        <Segmented
          value={semantic}
          onChange={setSemantic}
          options={[
            { value: "", label: "Any" },
            { value: "palm", label: "Palm" },
            { value: "ambiguous", label: "Ambiguous" },
            { value: "not_reviewed", label: "Not reviewed" },
          ]}
        />
      </div>
      <div className="row" style={{ alignItems: "flex-start", gap: 18 }}>
        <div>
          <span className="field-label">YOLO confidence</span>
          <div className="chips">
            {QUINTILES.map((x) => (
              <button key={x} className={`chip${confQ === x ? " on" : ""}`} onClick={() => setConfQ(confQ === x ? null : x)} title={`${x} of 5 (Q1 = lowest)`}>
                {x}
              </button>
            ))}
          </div>
        </div>
        <div>
          <span className="field-label">Box size</span>
          <div className="chips">
            {QUINTILES.map((x) => (
              <button key={x} className={`chip${areaQ === x ? " on" : ""}`} onClick={() => setAreaQ(areaQ === x ? null : x)} title={`${x} of 5 (Q1 = smallest)`}>
                {x}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div>
        <span className="field-label">A1 and A5 decisions differ for</span>
        <select className="input" value={changed} onChange={(e) => setChanged(e.target.value)}>
          <option value="">— any detection —</option>
          {changeModels.map((m) => (
            <option key={m.model_key} value={m.model_key}>
              {m.display}
            </option>
          ))}
        </select>
      </div>
      {categories.length > 0 && (
        <div>
          <span className="field-label">Curated cases (analysis a09)</span>
          <select
            className="input"
            value={curatedCat}
            onChange={(e) => {
              setCuratedCat(e.target.value);
              const first = curated.data?.find((c) => c.category === e.target.value);
              if (first) onSelect(first.sample_id);
            }}
          >
            <option value="">— choose a category —</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {humanize(c)}
              </option>
            ))}
          </select>
          {curatedCat && (
            <div className="chips" style={{ marginTop: 6 }}>
              {curated.data
                ?.filter((c) => c.category === curatedCat)
                .map((c) => (
                  <button key={c.sample_id} className={`chip${selected === c.sample_id ? " on" : ""}`} onClick={() => onSelect(c.sample_id)}>
                    {c.sample_id.replace("sample_", "")}
                  </button>
                ))}
            </div>
          )}
        </div>
      )}

      <div className="row small muted">
        <span>{list.data ? `${num(list.data.total)} detections` : "…"}</span>
        <span className="spacer" />
        <button className="chip" disabled={page <= 1} onClick={() => setPage(page - 1)}>‹</button>
        <span className="tiny">{page} / {pages}</span>
        <button className="chip" disabled={page >= pages} onClick={() => setPage(page + 1)}>›</button>
      </div>
      <div className="browser-list">
        {list.error && <div className="error">{list.error}</div>}
        {list.data?.items.map((d) => (
          <div key={d.detection_id} className={`browser-item${selected === d.detection_id ? " on" : ""}`} onClick={() => onSelect(d.detection_id)}>
            <span className="id">{d.detection_id}</span>
            <span>{d.semantic_label ? <SemanticBadge label={d.semantic_label} /> : null}</span>
            <span className="meta">
              <span>conf {d.yolo_confidence.toFixed(2)}</span>
              <span>size {d.area_quintile}</span>
              <span>{d.labelme_matched ? `IoU ${d.max_iou?.toFixed(2)}` : `unmatched · IoU ${d.max_iou?.toFixed(2)}`}</span>
            </span>
          </div>
        ))}
      </div>
    </aside>
  );
}
