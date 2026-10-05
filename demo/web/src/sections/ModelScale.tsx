import { useState } from "react";
import { api } from "../api/client";
import { LabelKey, StackBar, TransitionMatrix } from "../components/charts/Bars";
import { LadderChart } from "../components/charts/LadderChart";
import { ErrorNote, Loading, Section, Sources } from "../components/common/Section";
import { useApi } from "../hooks/useApi";
import type { LadderCheckpoint, Scaling } from "../types/api";
import { ci, fixed, humanize, pct } from "../utils/format";

export function ModelScale() {
  const data = useApi<Scaling>(() => api.getScaling(), []);
  const [ladder, setLadder] = useState("Qwen3-VL");
  const lad = data.data?.ladders.find((l) => l.ladder === ladder) ?? data.data?.ladders[0];

  return (
    <Section
      id="scale"
      kicker="04 · Model Scale"
      title="Same family, different size, different behaviour"
      lede={
        <>
          Within-family size ladders, A1 (overlay) versus A5 (crop only). Checkpoints are shown in size order with their real
          status: pending runs and degenerate checkpoints stay visible. A few points per family describe an ordering; they are
          not evidence for a general scaling law.
        </>
      }
    >
      {data.error && <ErrorNote error={data.error} />}
      {data.loading && !data.data && <Loading />}
      {data.data && lad && (
        <>
          <div className="row" style={{ marginBottom: 22 }}>
            <div className="seg">
              {data.data.ladders.map((l) => (
                <button key={l.ladder} className={l.ladder === lad.ladder ? "on" : ""} onClick={() => setLadder(l.ladder)}>
                  {l.ladder}
                </button>
              ))}
            </div>
            <span className="spacer" />
            <LabelKey />
          </div>
          <div className="ladder">
            <div className="ladder-grid" style={{ gridTemplateColumns: `repeat(${lad.checkpoints.length}, 1fr)` }}>
              {lad.checkpoints.map((c) => (
                <Checkpoint key={c.model_key} c={c} />
              ))}
            </div>
          </div>
          <div className="cols-2">
            <div>
              <h4 style={{ fontSize: 18, marginBottom: 8 }}>A1 → A5 change by checkpoint</h4>
              <LadderChart checkpoints={lad.checkpoints} />
              <p className="tiny muted">Whiskers: 95% image-level cluster bootstrap intervals (2,000 replicates). Hollow points: single-label checkpoint.</p>
            </div>
            <div>
              <h4 style={{ fontSize: 18, marginBottom: 8 }}>Observed orderings</h4>
              <table className="data">
                <thead>
                  <tr>
                    <th>Quantity</th>
                    <th>Values (size order)</th>
                    <th>Pattern</th>
                  </tr>
                </thead>
                <tbody>
                  {lad.orderings.map((o) => (
                    <tr key={o.quantity}>
                      <td className="small">{humanize(o.quantity.replace(/_A1_to_A5$/, "").replace(/_rate$/, ""))}</td>
                      <td className="mono tiny">{o.values.split(";").map((v) => pct(Number(v), 0)).join(" → ")}</td>
                      <td className="small">{o.observed_pattern}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {lad.orderings[0] && (
                <p className="note small" style={{ marginTop: 14 }}>
                  {lad.orderings[0].note}.
                  {lad.orderings[0].excluded_single_label && <> Excluded as single-label: {lad.orderings[0].excluded_single_label}.</>}
                  {lad.orderings[0].not_available && <> Not available: {lad.orderings[0].not_available}.</>}
                </p>
              )}
            </div>
          </div>
        </>
      )}
      <Sources paths={data.data?.sources ?? []} />
    </Section>
  );
}

function Checkpoint({ c }: { c: LadderCheckpoint }) {
  const na = c.changed === null || c.changed === undefined;
  if (na) {
    return (
      <div className="ladder-col na">
        <h4>{c.model}</h4>
        <div className="size">{c.nominal_b ? `${c.nominal_b}B nominal` : ""}</div>
        <p className="small" style={{ marginTop: 14 }}>{humanize(c.status)}</p>
        {c.note && <p className="tiny">{c.note}</p>}
        <p className="tiny">Displayed only when the analysis layer audits its results as usable.</p>
      </div>
    );
  }
  const a1 = c.label_usage?.find((u) => u.condition === "A1");
  const a5 = c.label_usage?.find((u) => u.condition === "A5");
  const collapsed = c.single_label_A1_and_A5 || (a1 && a1["diag_majority_ge_0.99"]) || (a5 && a5["diag_majority_ge_0.99"]);
  return (
    <div className="ladder-col">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h4>{c.model}</h4>
        {c.single_label_A1_and_A5 ? <span className="flag">single label</span> : collapsed ? <span className="flag">near-collapse</span> : null}
      </div>
      <div className="size">{c.params_billions_counted ? `${c.params_billions_counted}B params (counted)` : ""}</div>
      {c.status !== "five conditions usable" && <div className="tiny muted">{c.status}</div>}
      <div className="mini-k"><span>A1 overlay</span><span>H = {fixed(c.A1_entropy, 2)} bits</span></div>
      <StackBar rates={{ Reliable: c.A1_R, Uncertain: c.A1_U, Unreliable: c.A1_Ur }} />
      <div className="mini-k"><span>A5 crop only</span><span>H = {fixed(c.A5_entropy, 2)} bits</span></div>
      <StackBar rates={{ Reliable: c.A5_R, Uncertain: c.A5_U, Unreliable: c.A5_Ur }} />
      <table className="data" style={{ marginTop: 12, fontSize: 12.5 }}>
        <tbody>
          <tr><td>changed</td><td className="num">{pct(c.changed)}</td><td className="num tiny muted">{ci(c.changed_ci_low, c.changed_ci_high)}</td></tr>
          <tr><td style={{ color: "#7e5c16" }}>into U</td><td className="num">{pct(c.into_U)}</td><td className="num tiny muted">{ci(c.into_U_ci_low, c.into_U_ci_high)}</td></tr>
          <tr><td style={{ color: "var(--unr)" }}>into Ur</td><td className="num">{pct(c.into_Ur)}</td><td className="num tiny muted">{ci(c.into_Ur_ci_low, c.into_Ur_ci_high)}</td></tr>
        </tbody>
      </table>
      {c.matrix && c.matrix.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <TransitionMatrix cells={c.matrix} fromLabel="A1" toLabel="A5" />
        </div>
      )}
    </div>
  );
}
