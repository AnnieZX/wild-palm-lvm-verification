import { useMemo, useState } from "react";
import { api } from "../api/client";
import { Verdict } from "../components/common/Labels";
import { Segmented } from "../components/common/Segmented";
import { ErrorNote, Loading, Section, Sources } from "../components/common/Section";
import { useApi } from "../hooks/useApi";
import type { Difficulty as D, DifficultyModel, Quintile } from "../types/api";
import { fixed, pct } from "../utils/format";

type Prop = "confidence" | "box_area";
type Ev = "changed" | "toward_rejection" | "into_U" | "into_Ur";

const EVENT_TEXT: Record<Ev, string> = {
  changed: "label changed",
  toward_rejection: "moved toward rejection",
  into_U: "moved into Uncertain",
  into_Ur: "moved into Unreliable",
};
const EVENT_COLOR: Record<Ev, string> = { changed: "var(--ink)", toward_rejection: "var(--ink-2)", into_U: "var(--unc)", into_Ur: "var(--unr)" };
const Q: Quintile[] = ["Q1", "Q2", "Q3", "Q4", "Q5"];
const ORDER = ["supports", "reverses", "null", "not"];

export function Difficulty() {
  const [pair, setPair] = useState("A1->A5");
  const [prop, setProp] = useState<Prop>("box_area");
  const [event, setEvent] = useState<Ev>("changed");
  const data = useApi<D>(() => api.getDifficulty(pair), [pair]);

  const models = useMemo(() => {
    const list = (data.data?.models ?? []).map((m) => ({
      m,
      contrast: m.contrasts.find((c) => c.property === prop && c.event === event),
    }));
    return list.sort((a, b) => {
      const va = ORDER.findIndex((o) => (a.contrast?.verdict ?? "not").startsWith(o));
      const vb = ORDER.findIndex((o) => (b.contrast?.verdict ?? "not").startsWith(o));
      return va - vb || (b.contrast?.diff_Q1_minus_Q5 ?? 0) - (a.contrast?.diff_Q1_minus_Q5 ?? 0);
    });
  }, [data.data, prop, event]);

  const counts = useMemo(() => {
    const c = { supports: 0, reverses: 0, null: 0, not: 0 };
    models.forEach(({ contrast }) => {
      const v = contrast?.verdict ?? "not";
      if (v.startsWith("supports")) c.supports++;
      else if (v.startsWith("reverses")) c.reverses++;
      else if (v.startsWith("null")) c.null++;
      else c.not++;
    });
    return c;
  }, [models]);

  const ymax = useMemo(() => {
    let mx = 0.1;
    models.forEach(({ m }) => m.rates.forEach((r) => r.property === prop && r.event === event && (mx = Math.max(mx, r.ci_high ?? r.rate))));
    return Math.min(1, Math.ceil(mx * 10) / 10);
  }, [models, prop, event]);

  const info = data.data?.property_info;
  const propName = prop === "box_area" ? "box size" : "YOLO confidence";

  return (
    <Section
      id="difficulty"
      kicker="05 · Detection Difficulty"
      title="Which detections are most sensitive to losing context?"
      lede={
        <>
          Detections are split into cohort quintiles of normalised box area and of YOLO confidence (Q1 = smallest / lowest). For
          many checkpoints, Q1 detections change more when context is removed, but not for all. Checkpoints that reverse the
          pattern are shown directly after those that support it.
        </>
      }
    >
      <div className="row" style={{ gap: 28, marginBottom: 20, alignItems: "flex-end" }}>
        <div>
          <span className="field-label">Property</span>
          <Segmented<Prop> value={prop} onChange={setProp} options={[{ value: "box_area", label: "Box size" }, { value: "confidence", label: "YOLO confidence" }]} />
        </div>
        <div>
          <span className="field-label">Event</span>
          <Segmented<Ev> value={event} onChange={setEvent} options={(Object.keys(EVENT_TEXT) as Ev[]).map((e) => ({ value: e, label: EVENT_TEXT[e] }))} />
        </div>
        <div>
          <span className="field-label">Condition pair</span>
          <Segmented value={pair} onChange={setPair} options={(data.data?.pairs ?? ["A1->A5", "A2->A5"]).map((p) => ({ value: p, label: p.replace("->", " → ") }))} />
        </div>
      </div>
      {data.error && <ErrorNote error={data.error} />}
      {data.loading && !data.data && <Loading />}
      {data.data && (
        <>
          <p style={{ fontSize: 17, maxWidth: "var(--measure)" }}>
            <strong>{counts.supports}</strong> checkpoints: the {prop === "box_area" ? "smallest" : "lowest-confidence"} quintile {EVENT_TEXT[event].replace("label ", "")} more often
            than the {prop === "box_area" ? "largest" : "highest-confidence"}. <strong>{counts.reverses}</strong> show the opposite,{" "}
            <strong>{counts.null}</strong> show no clear difference{counts.not ? <>, and <strong>{counts.not}</strong> cannot be evaluated (single label)</> : null}.
          </p>
          <div className="smallmult" style={{ marginTop: 28 }}>
            {models.map(({ m, contrast }) => (
              <Mini key={m.model} m={m} prop={prop} event={event} ymax={ymax} contrast={contrast} />
            ))}
          </div>
          {info && (
            <p className="note warn small" style={{ marginTop: 36 }}>
              Box size and YOLO confidence are correlated in this cohort (Spearman ρ = {fixed(info.spearman_confidence_vs_norm_area, 2)}, N ={" "}
              {info.n.toLocaleString()}), so the two patterns are not independent evidence. These are descriptive quintile
              contrasts with bootstrap intervals, not causal estimates. Currently showing {propName}.
            </p>
          )}
        </>
      )}
      <Sources paths={data.data?.sources ?? []} />
    </Section>
  );
}

function Mini({
  m,
  prop,
  event,
  ymax,
  contrast,
}: {
  m: DifficultyModel;
  prop: Prop;
  event: Ev;
  ymax: number;
  contrast: DifficultyModel["contrasts"][number] | undefined;
}) {
  const W = 240;
  const H = 120;
  const pad = { l: 28, r: 6, t: 8, b: 20 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const rates = Q.map((q) => m.rates.find((r) => r.property === prop && r.event === event && r.quintile === q));
  const x = (i: number) => pad.l + (i / 4) * iw;
  const y = (v: number) => pad.t + ih - (v / ymax) * ih;
  const pts = rates.map((r, i) => (r ? `${x(i)},${y(r.rate)}` : null)).filter(Boolean);
  const band =
    rates.every((r) => r && r.ci_low !== null && r.ci_high !== null) && rates.length
      ? `M${rates.map((r, i) => `${x(i)},${y(r!.ci_high!)}`).join("L")}L${[...rates].reverse().map((r, j) => `${x(4 - j)},${y(r!.ci_low!)}`).join("L")}Z`
      : null;
  const color = EVENT_COLOR[event];
  return (
    <div className="sm-cell">
      <h5>
        <span>{m.model}</span>
        <Verdict value={contrast?.verdict} />
      </h5>
      <div className="tiny muted">
        Q1 {pct(contrast?.rate_Q1, 0)} · Q5 {pct(contrast?.rate_Q5, 0)} · Δ {contrast ? `${(contrast.diff_Q1_minus_Q5 * 100).toFixed(1)} pp` : "—"}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {[0, ymax / 2, ymax].map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--rule)" />
            <text x={pad.l - 4} y={y(t) + 3.5} fontSize={9.5} textAnchor="end" fill="var(--ink-3)">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {band && <path d={band} fill={color} opacity={0.12} />}
        <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={1.6} />
        {rates.map((r, i) => r && <circle key={i} cx={x(i)} cy={y(r.rate)} r={2.6} fill={color}><title>{`${Q[i]}: ${pct(r.rate)} (${r.n_event}/${r.n_at_risk})`}</title></circle>)}
        {Q.map((q, i) => (
          <text key={q} x={x(i)} y={H - 5} fontSize={9.5} textAnchor="middle" fill="var(--ink-3)">
            {q}
          </text>
        ))}
      </svg>
    </div>
  );
}
