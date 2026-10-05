import { useMemo, useState } from "react";
import type { Condition, PredictionRow, Predictions } from "../../types/api";
import { CONDITIONS } from "../../utils/research";
import { CellStateMark, DegeneracyFlag, VlmBadge } from "../common/Labels";

interface Props {
  predictions: Predictions;
  condition: Condition;
  onCondition: (c: Condition) => void;
  picked: { model: string; condition: Condition } | null;
  onPick: (model: string, condition: Condition) => void;
}

export function PredictionMatrix({ predictions, condition, onCondition, picked, onPick }: Props) {
  const families = useMemo(() => Array.from(new Set(predictions.models.map((m) => m.family))), [predictions]);
  const [family, setFamily] = useState<string>("");
  const [hideEmpty, setHideEmpty] = useState(false);

  const rows = predictions.models.filter(
    (m) => (!family || m.family === family) && (!hideEmpty || CONDITIONS.some((c) => m.cells[c].state === "decision")),
  );
  const grouped = families
    .filter((f) => !family || f === family)
    .map((f) => ({ family: f, rows: rows.filter((r) => r.family === f) }))
    .filter((g) => g.rows.length);

  const nChanging = predictions.models.filter((m) => m.changes_across_conditions).length;
  const nWithAny = predictions.models.filter((m) => CONDITIONS.some((c) => m.cells[c].state === "decision")).length;

  return (
    <div>
      <div className="row" style={{ marginBottom: 10 }}>
        <span className="field-label" style={{ margin: 0 }}>
          Family
        </span>
        <div className="chips">
          <button className={`chip${!family ? " on" : ""}`} onClick={() => setFamily("")}>
            All
          </button>
          {families.map((f) => (
            <button key={f} className={`chip${family === f ? " on" : ""}`} onClick={() => setFamily(f)}>
              {f}
            </button>
          ))}
        </div>
        <span className="spacer" />
        <label className="small muted" style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
          <input type="checkbox" checked={hideEmpty} onChange={(e) => setHideEmpty(e.target.checked)} />
          hide checkpoints without results
        </label>
      </div>
      <table className="data matrix">
        <thead>
          <tr>
            <th>Checkpoint</th>
            <th className="num">Params</th>
            {CONDITIONS.map((c) => (
              <th key={c} className={`cond${c === condition ? " on" : ""}`} onClick={() => onCondition(c)}>
                {c}
              </th>
            ))}
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {grouped.map((g) => (
            <Group key={g.family} family={g.family} rows={g.rows} condition={condition} picked={picked} onPick={onPick} />
          ))}
        </tbody>
      </table>
      <div className="row small muted" style={{ marginTop: 10, gap: 20 }}>
        <span>
          <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "var(--ink)", marginRight: 6 }} />
          decision differs from the same checkpoint’s A1 decision
        </span>
        <span>
          {nChanging} of {nWithAny} checkpoints with results change their decision somewhere across A1–A5 for this detection.
        </span>
      </div>
    </div>
  );
}

function Group({
  family,
  rows,
  condition,
  picked,
  onPick,
}: {
  family: string;
  rows: PredictionRow[];
  condition: Condition;
  picked: Props["picked"];
  onPick: Props["onPick"];
}) {
  return (
    <>
      <tr className="group">
        <td colSpan={8}>{family}</td>
      </tr>
      {rows.map((r) => {
        const a1 = r.cells.A1.decision;
        return (
          <tr key={r.model_key}>
            <td className="model">{r.display}</td>
            <td className="num muted tiny">{r.params_billions ? `${r.params_billions.toFixed(1)}B` : "—"}</td>
            {CONDITIONS.map((c) => {
              const cell = r.cells[c];
              const changed = a1 && cell.decision && cell.decision !== a1 && c !== "A1";
              const isPicked = picked?.model === r.model_key && picked.condition === c;
              return (
                <td
                  key={c}
                  className={`cell${c === condition ? " on" : ""}${isPicked ? " picked" : ""}`}
                  onClick={() => onPick(r.model_key, c)}
                  title="Show this checkpoint’s stored response"
                >
                  {cell.state === "decision" && cell.decision ? <VlmBadge label={cell.decision} /> : <CellStateMark state={cell.state} />}
                  {changed && <i className="chg" />}
                </td>
              );
            })}
            <td>
              <DegeneracyFlag value={r.degenerate} />
            </td>
          </tr>
        );
      })}
    </>
  );
}
