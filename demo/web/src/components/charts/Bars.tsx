import type { Distribution, TransitionCell, VlmLabel } from "../../types/api";
import { pct } from "../../utils/format";
import { VLM_LABELS, VLM_SHORT } from "../../utils/research";

/** Horizontal stacked R/U/Ur bar from rates (0..1). */
export function StackBar({ rates, title }: { rates: Partial<Record<VlmLabel, number | null>>; title?: string }) {
  const tip =
    title ?? VLM_LABELS.map((l) => `${l} ${pct(rates[l] ?? null)}`).join(" · ");
  return (
    <div className="stackbar" title={tip}>
      {VLM_LABELS.map((l) => {
        const v = rates[l] ?? 0;
        return v > 0 ? <span key={l} className={l} style={{ width: `${v * 100}%` }} /> : null;
      })}
    </div>
  );
}

export function distRates(d: Distribution | undefined): Partial<Record<VlmLabel, number>> {
  const out: Partial<Record<VlmLabel, number>> = {};
  if (!d) return out;
  for (const l of VLM_LABELS) out[l] = d[l]?.rate ?? 0;
  return out;
}

/** Single-value bar with optional CI whisker; `max` sets the full-scale value. */
export function HBar({
  value,
  max = 1,
  kind,
  lo,
  hi,
}: {
  value: number | null | undefined;
  max?: number;
  kind?: "U" | "Ur" | "R";
  lo?: number | null;
  hi?: number | null;
}) {
  if (value === null || value === undefined) return <div className="hbar" title="not available" />;
  const w = Math.min(1, value / max) * 100;
  return (
    <div className={`hbar ${kind ?? ""}`} title={pct(value)}>
      <span style={{ width: `${w}%` }} />
      {lo !== null && lo !== undefined && hi !== null && hi !== undefined && (
        <i
          className="ci"
          style={{ left: `${(lo / max) * 100}%`, width: `${Math.max(0.5, ((hi - lo) / max) * 100)}%` }}
        />
      )}
    </div>
  );
}

const CELL_BG: Record<VlmLabel, string> = { Reliable: "47,106,82", Uncertain: "178,132,43", Unreliable: "163,68,47" };

/** 3x3 transition matrix; shading is the share of all paired detections, coloured by destination label. */
export function TransitionMatrix({
  cells,
  fromLabel,
  toLabel,
}: {
  cells: TransitionCell[];
  fromLabel: string;
  toLabel: string;
}) {
  const total = cells.reduce((s, c) => s + c.count, 0) || 1;
  const get = (f: VlmLabel, t: VlmLabel) => cells.find((c) => c.from === f && c.to === t)?.count ?? 0;
  return (
    <div>
      <div className="tmatrix">
        <div className="h tiny">
          {fromLabel} ↓ / {toLabel} →
        </div>
        {VLM_LABELS.map((t) => (
          <div key={t} className="h col">
            {VLM_SHORT[t]}
          </div>
        ))}
        {VLM_LABELS.map((f) => (
          <Row key={f} f={f} get={get} total={total} />
        ))}
      </div>
    </div>
  );
}

function Row({ f, get, total }: { f: VlmLabel; get: (f: VlmLabel, t: VlmLabel) => number; total: number }) {
  return (
    <>
      <div className="h">{f}</div>
      {VLM_LABELS.map((t) => {
        const n = get(f, t);
        const share = n / total;
        const alpha = n === 0 ? 0 : 0.08 + Math.min(1, share * 2.2) * 0.82;
        const dark = alpha > 0.5;
        return (
          <div
            key={t}
            className={`c${f === t ? " diag" : ""}`}
            style={{ background: `rgba(${CELL_BG[t]},${alpha})`, color: dark ? "#fff" : "var(--ink-2)" }}
            title={`${f} → ${t}: ${n.toLocaleString()} (${pct(share)})`}
          >
            {n ? pct(share, share < 0.01 ? 1 : 0) : "·"}
          </div>
        );
      })}
    </>
  );
}

export function LabelKey() {
  return (
    <div className="key">
      <span><i style={{ background: "var(--rel)" }} />Reliable</span>
      <span><i style={{ background: "var(--unc)" }} />Uncertain</span>
      <span><i style={{ background: "var(--unr)" }} />Unreliable</span>
    </div>
  );
}
