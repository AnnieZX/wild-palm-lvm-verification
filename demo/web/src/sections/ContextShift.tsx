import { useMemo, useState } from "react";
import { api } from "../api/client";
import { HBar, LabelKey, StackBar, TransitionMatrix, distRates } from "../components/charts/Bars";
import { DegeneracyFlag } from "../components/common/Labels";
import { ErrorNote, Loading, Section, Sources } from "../components/common/Section";
import { useApi } from "../hooks/useApi";
import type { ContextShift as CS, ModelInfo } from "../types/api";
import { num, pct } from "../utils/format";
import { PAIR_GROUP_TEXT } from "../utils/research";

type SortKey = "changed_rate" | "into_U_rate" | "into_Ur_rate" | "registry";

export function ContextShift({ models }: { models: ModelInfo[] }) {
  const [pair, setPair] = useState("A1->A5");
  const [sort, setSort] = useState<SortKey>("changed_rate");
  const [sel, setSel] = useState<string | null>(null);
  const data = useApi<CS>(() => api.getContextShift(pair), [pair]);
  const byKey = useMemo(() => Object.fromEntries(models.map((m) => [m.model_key, m])), [models]);
  const [from, to] = pair.split("->");

  const rows = useMemo(() => {
    const list = [...(data.data?.models ?? [])];
    if (sort === "registry") list.sort((a, b) => (byKey[a.model_key ?? ""]?.registry_order ?? 99) - (byKey[b.model_key ?? ""]?.registry_order ?? 99));
    else list.sort((a, b) => ((b[sort] as number) ?? -1) - ((a[sort] as number) ?? -1));
    return list;
  }, [data.data, sort, byKey]);
  const selected = rows.find((r) => r.model === sel) ?? rows[0];
  const pairInfo = data.data?.pairs.find((p) => p.pair === pair);
  const groups = Array.from(new Set((data.data?.pairs ?? []).map((p) => p.group)));
  const absent = models.filter((m) => !rows.some((r) => r.model_key === m.model_key));

  return (
    <Section
      id="context-shift"
      kicker="03 · Context Shift"
      title="Across 5,747 detections, how do decisions move?"
      lede={
        <>
          For each checkpoint, the share of detections whose label changes between two conditions and where those changes go.
          A move toward Uncertain or Unreliable is a change in behaviour; without a semantic reference standard for the
          matched detections, neither direction is “correct” by itself.
        </>
      }
    >
      <div className="row" style={{ marginBottom: 18, alignItems: "flex-end", gap: 28 }}>
        {groups.map((g) => (
          <div key={g}>
            <span className="field-label">{PAIR_GROUP_TEXT[g] ?? g}</span>
            <div className="chips">
              {data.data?.pairs
                .filter((p) => p.group === g)
                .map((p) => (
                  <button key={p.pair} className={`chip${p.pair === pair ? " on" : ""}`} onClick={() => setPair(p.pair)}>
                    {p.pair.replace("->", " → ")}
                  </button>
                ))}
            </div>
          </div>
        ))}
      </div>
      {pairInfo?.note && <p className="note small" style={{ marginBottom: 24 }}>{pairInfo.note}</p>}
      {data.error && <ErrorNote error={data.error} />}
      {data.loading && !data.data && <Loading />}
      {data.data && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 40, alignItems: "start" }}>
          <div>
            <div className="row" style={{ marginBottom: 8 }}>
              <LabelKey />
              <span className="spacer" />
              <span className="field-label" style={{ margin: 0 }}>sort</span>
              <div className="chips">
                {(
                  [
                    ["changed_rate", "changed"],
                    ["into_U_rate", "→ Uncertain"],
                    ["into_Ur_rate", "→ Unreliable"],
                    ["registry", "registry"],
                  ] as [SortKey, string][]
                ).map(([k, l]) => (
                  <button key={k} className={`chip${sort === k ? " on" : ""}`} onClick={() => setSort(k)}>
                    {l}
                  </button>
                ))}
              </div>
            </div>
            <table className="data">
              <thead>
                <tr>
                  <th>Checkpoint</th>
                  <th style={{ width: "15%" }}>{from} labels</th>
                  <th style={{ width: "15%" }}>{to} labels</th>
                  <th className="num">changed</th>
                  <th style={{ width: "12%" }} />
                  <th className="num">into U</th>
                  <th className="num">into Ur</th>
                  <th className="num">into R</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.model} className={`clickable${selected?.model === r.model ? " selected" : ""}`} onClick={() => setSel(r.model)}>
                    <td>
                      {r.model}
                      {r.pair_status !== "full_clean" && <span className="muted" title={`pair status: ${r.pair_status}; N = ${r.n}`}> †</span>}
                    </td>
                    <td><StackBar rates={distRates(r.distributions[from])} /></td>
                    <td><StackBar rates={distRates(r.distributions[to])} /></td>
                    <td className="num">{pct(r.changed_rate)}</td>
                    <td><HBar value={r.changed_rate} /></td>
                    <td className="num" style={{ color: "#7e5c16" }}>{pct(r.into_U_rate)}</td>
                    <td className="num" style={{ color: "var(--unr)" }}>{pct(r.into_Ur_rate)}</td>
                    <td className="num" style={{ color: "var(--rel)" }}>{pct(r.into_R_rate)}</td>
                    <td><DegeneracyFlag value={byKey[r.model_key ?? ""]?.degenerate ?? null} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="tiny muted" style={{ marginTop: 10 }}>
              “into U” = share of all paired detections whose label changed to Uncertain (R→U + Ur→U); likewise for Ur and R.
              † partial cells: pairs use detections with a valid decision in both conditions.
              {absent.length > 0 && <> Not shown (fewer than two usable conditions for this pair): {absent.map((m) => m.display).join(", ")}.</>}
            </p>
          </div>
          {selected && (
            <aside style={{ position: "sticky", top: 70 }}>
              <div className="field-label">Transition matrix</div>
              <h4 style={{ fontSize: 20, marginBottom: 4 }}>{selected.model}</h4>
              <p className="small muted">
                {pair.replace("->", " → ")} · N = {num(selected.n)} paired detections · {num(selected.n_changed)} changed
              </p>
              <TransitionMatrix cells={selected.matrix} fromLabel={from} toLabel={to} />
              <p className="tiny muted" style={{ marginTop: 10 }}>
                Cells show the share of all paired detections. Diagonal = unchanged.
              </p>
            </aside>
          )}
        </div>
      )}
      <Sources paths={data.data?.sources ?? []} />
    </Section>
  );
}
