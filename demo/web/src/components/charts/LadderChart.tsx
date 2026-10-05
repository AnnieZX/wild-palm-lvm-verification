import type { LadderCheckpoint } from "../../types/api";
import { pct } from "../../utils/format";

const SERIES = [
  { key: "changed", lo: "changed_ci_low", hi: "changed_ci_high", label: "changed", color: "var(--ink)" },
  { key: "into_U", lo: "into_U_ci_low", hi: "into_U_ci_high", label: "into Uncertain", color: "var(--unc)" },
  { key: "into_Ur", lo: "into_Ur_ci_low", hi: "into_Ur_ci_high", label: "into Unreliable", color: "var(--unr)" },
] as const;

/** A1→A5 rates across checkpoints in size order; whiskers are image-level bootstrap 95% intervals. */
export function LadderChart({ checkpoints }: { checkpoints: LadderCheckpoint[] }) {
  const W = 620;
  const H = 230;
  const m = { l: 40, r: 16, t: 14, b: 40 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const n = checkpoints.length;
  const x = (i: number) => m.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v: number) => m.t + ih - v * ih;
  const val = (c: LadderCheckpoint, k: string) => (c as unknown as Record<string, number | null>)[k];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="A1 to A5 rates by checkpoint size">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke="var(--rule)" />
          <text x={m.l - 8} y={y(t) + 4} fontSize={10.5} textAnchor="end" fill="var(--ink-3)">
            {t * 100}%
          </text>
        </g>
      ))}
      {checkpoints.map((c, i) => {
        const na = c.changed === null || c.changed === undefined;
        return (
          <g key={c.model_key}>
            {na && <rect x={x(i) - 26} y={m.t} width={52} height={ih} fill="url(#hatch)" />}
            <text x={x(i)} y={H - 22} fontSize={11.5} textAnchor="middle" fill="var(--ink)" fontWeight={600}>
              {c.model.replace(/^.*-(\d+B)$/, "$1")}
            </text>
            <text x={x(i)} y={H - 8} fontSize={10} textAnchor="middle" fill="var(--ink-3)">
              {na ? (c.status.startsWith("pending") ? "pending" : "not run") : c.single_label_A1_and_A5 ? "single label" : `${c.params_billions_counted ?? ""}B params`}
            </text>
          </g>
        );
      })}
      <defs>
        <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--rule)" strokeWidth="3" />
        </pattern>
      </defs>
      {SERIES.map((s, si) => {
        const pts = checkpoints.map((c, i) => ({ i, v: val(c, s.key), lo: val(c, s.lo), hi: val(c, s.hi), single: c.single_label_A1_and_A5 }));
        const segs: string[] = [];
        for (let k = 1; k < pts.length; k++) {
          const a = pts[k - 1];
          const b = pts[k];
          if (a.v !== null && a.v !== undefined && b.v !== null && b.v !== undefined) segs.push(`M${x(a.i) + (si - 1) * 4},${y(a.v)}L${x(b.i) + (si - 1) * 4},${y(b.v)}`);
        }
        return (
          <g key={s.key}>
            <path d={segs.join("")} stroke={s.color} strokeWidth={1.5} fill="none" opacity={0.8} />
            {pts.map((p) =>
              p.v === null || p.v === undefined ? null : (
                <g key={p.i}>
                  {p.lo !== null && p.hi !== null && p.lo !== undefined && p.hi !== undefined && (
                    <line x1={x(p.i) + (si - 1) * 4} x2={x(p.i) + (si - 1) * 4} y1={y(p.lo)} y2={y(p.hi)} stroke={s.color} strokeWidth={1.2} />
                  )}
                  <circle cx={x(p.i) + (si - 1) * 4} cy={y(p.v)} r={3.6} fill={p.single ? "#fff" : s.color} stroke={s.color} strokeWidth={1.5}>
                    <title>{`${s.label}: ${pct(p.v)}`}</title>
                  </circle>
                </g>
              ),
            )}
          </g>
        );
      })}
      <g transform={`translate(${m.l + 6}, ${m.t + 4})`}>
        {SERIES.map((s, i) => (
          <g key={s.key} transform={`translate(${i * 120}, 0)`}>
            <circle cx={4} cy={4} r={3.5} fill={s.color} />
            <text x={12} y={8} fontSize={11} fill="var(--ink-2)">
              {s.label}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}
