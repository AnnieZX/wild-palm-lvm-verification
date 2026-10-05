import { useMemo, useState } from "react";
import type { Condition, ReviewResultItem, ReviewResults, VisitorLabel } from "../../types/api";
import { fixed } from "../../utils/format";
import { CONDITIONS, reviewCropWindow, SEMANTIC_LABELS, SEMANTIC_NAME, UNMATCHED_REASON_TEXT } from "../../utils/research";
import { DegeneracyFlag, SemanticBadge, VlmBadge } from "../common/Labels";
import { Segmented } from "../common/Segmented";
import { PatchView } from "../detection/PatchView";

const DEFAULT_MODELS = ["qwen3_vl", "internvl3_5_hf_4b", "gemma4", "glm_4_6v_flash"];

export function ReviewReveal({ results, onOpenDetection }: { results: ReviewResults; onOpenDetection: (id: string) => void }) {
  const [focus, setFocus] = useState(0);
  const item = results.items[focus];
  const answered = results.items.filter((i) => i.visitor_label !== "skip");
  const counts = (key: "visitor_label" | "research_label") =>
    Object.fromEntries(SEMANTIC_LABELS.map((l) => [l, results.items.filter((i) => i[key] === l).length]));
  const mine = counts("visitor_label");
  const theirs = counts("research_label");
  const skipped = results.items.length - answered.length;

  return (
    <div>
      <h3 style={{ fontSize: 26, marginBottom: 8 }}>Your review</h3>
      <p style={{ fontSize: 17, maxWidth: "var(--measure)" }}>
        You agreed with the research reviewer on <strong>{results.n_agree}</strong> of <strong>{results.n_answered}</strong> answered
        item{results.n_answered === 1 ? "" : "s"}
        {skipped ? ` (${skipped} skipped)` : ""}. You:{" "}
        {SEMANTIC_LABELS.map((l) => `${mine[l]} ${SEMANTIC_NAME[l].toLowerCase()}`).join(" · ")}. Research audit:{" "}
        {SEMANTIC_LABELS.map((l) => `${theirs[l]} ${SEMANTIC_NAME[l].toLowerCase()}`).join(" · ")}.
      </p>
      {results.deck === "unmatched" && (
        <p className="note" style={{ marginBottom: 28 }}>
          Every detection you just judged is <strong>IoU-unmatched</strong>: under Protocol v2 it counts as GT− because no LabelMe palm
          box overlaps it at IoU ≥ 0.5. The dashed boxes below are the LabelMe annotations you could not see.
        </p>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 36, alignItems: "start" }}>
        <table className="data">
          <thead>
            <tr>
              <th />
              <th>You</th>
              <th>Research audit</th>
              <th />
              <th>LabelMe geometry</th>
              <th className="num">YOLO conf.</th>
            </tr>
          </thead>
          <tbody>
            {results.items.map((it, i) => (
              <tr key={it.index} className={`clickable${i === focus ? " selected" : ""}`} onClick={() => setFocus(i)}>
                <td style={{ width: 64 }}>
                  <Thumb it={it} />
                </td>
                <td><SemanticBadge label={it.visitor_label} /></td>
                <td><SemanticBadge label={it.research_label} /></td>
                <td>
                  {it.agrees === null ? <span className="agree muted">—</span> : it.agrees ? <span className="agree yes">agree</span> : <span className="agree no">differ</span>}
                </td>
                <td className="small">
                  IoU {fixed(it.max_iou, 2)}
                  <div className="tiny muted">
                    {it.unmatched_reason ? UNMATCHED_REASON_TEXT[it.unmatched_reason] ?? it.unmatched_reason : it.labelme_matched === false ? "not aligned at IoU ≥ 0.5" : ""}
                  </div>
                </td>
                <td className="num small">{fixed(it.yolo_confidence, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <aside style={{ position: "sticky", top: 70 }}>
          <div className="imgframe">
            <PatchView
              imageUrl={item.image_url}
              width={item.image_width ?? 912}
              height={item.image_height ?? 912}
              target={item.bbox_xywh}
              labelme={item.labelme_palm_boxes_xywh}
            />
          </div>
          <div className="legend">
            <span><i className="swatch" style={{ borderColor: "#e0b800" }} /> YOLO detection</span>
            <span><i className="swatch dashed" style={{ borderColor: "#2aa7d6" }} /> LabelMe palm boxes ({item.labelme_palm_boxes_xywh.length})</span>
          </div>
          <p className="small" style={{ marginTop: 10 }}>
            <span className="mono">{item.detection_id ?? item.ref_id}</span> · patch {item.image_id}
            {item.confidence_bin && <> · confidence bin {item.confidence_bin}</>}
          </p>
          {item.detection_id && (
            <button className="btn" onClick={() => onOpenDetection(item.detection_id!)}>
              Open in Detection Explorer
            </button>
          )}
        </aside>
      </div>

      <YouVsVlm results={results} />
    </div>
  );
}

function Thumb({ it }: { it: ReviewResultItem }) {
  const w = it.image_width ?? 912;
  const h = it.image_height ?? 912;
  return (
    <div className="thumb" style={{ width: 56, height: 56 }}>
      <PatchView imageUrl={it.image_url} width={w} height={h} target={it.bbox_xywh} window={reviewCropWindow(it.bbox_xywh, w, h)} labelme={it.labelme_palm_boxes_xywh} />
    </div>
  );
}

function YouVsVlm({ results }: { results: ReviewResults }) {
  const [cond, setCond] = useState<Condition>("A1");
  const [chosen, setChosen] = useState<string[]>(() => DEFAULT_MODELS.filter((k) => results.models.some((m) => m.model_key === k)));
  const models = results.models.filter((m) => chosen.includes(m.model_key));
  const hasVlm = results.items.some((i) => i.vlm);
  const toggle = (k: string) => setChosen((c) => (c.includes(k) ? c.filter((x) => x !== k) : c.length >= 6 ? c : [...c, k]));

  const tally = useMemo(() => {
    const t: Record<string, Record<string, number>> = {};
    for (const m of models) {
      t[m.model_key] = { Reliable: 0, Uncertain: 0, Unreliable: 0, none: 0 };
      for (const it of results.items) {
        const d = it.vlm?.[m.model_key]?.[cond];
        t[m.model_key][d ?? "none"]++;
      }
    }
    return t;
  }, [models, results.items, cond]);

  return (
    <div style={{ marginTop: 56 }}>
      <h3 style={{ fontSize: 26, marginBottom: 8 }}>You vs VLM</h3>
      <p style={{ maxWidth: "var(--measure)" }}>
        Your answer and the audit label say whether the region <em>is a palm</em>. A VLM’s Reliable / Uncertain / Unreliable says
        whether it judges <em>the detection</em> trustworthy under the input it was given. They are related but not the same
        question: a palm can be called Uncertain, and Unreliable also covers boxes the model considers badly placed. This is not
        an accuracy leaderboard; look for where the columns disagree and how that changes between conditions.
      </p>
      {!hasVlm ? (
        <p className="note warn">The lower-confidence pilot lies outside the 5,747 cohort; no VLM was run on these detections.</p>
      ) : (
        <>
          <div className="row" style={{ margin: "16px 0 12px", alignItems: "flex-end", gap: 24 }}>
            <div>
              <span className="field-label">Condition</span>
              <Segmented<Condition> value={cond} onChange={setCond} options={CONDITIONS.map((c) => ({ value: c, label: c }))} />
            </div>
            <div style={{ flex: 1 }}>
              <span className="field-label">Checkpoints (up to 6)</span>
              <div className="chips">
                {results.models.map((m) => (
                  <button key={m.model_key} className={`chip${chosen.includes(m.model_key) ? " on" : ""}`} onClick={() => toggle(m.model_key)}>
                    {m.display}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>Detection</th>
                <th>You</th>
                <th>Human audit</th>
                {models.map((m) => (
                  <th key={m.model_key}>
                    {m.display} <DegeneracyFlag value={m.degenerate} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.items.map((it) => (
                <tr key={it.index}>
                  <td className="mono small">{it.detection_id ?? it.ref_id}</td>
                  <td><SemanticBadge label={it.visitor_label as VisitorLabel} /></td>
                  <td><SemanticBadge label={it.research_label} /></td>
                  {models.map((m) => {
                    const d = it.vlm?.[m.model_key]?.[cond];
                    const usable = m.usable_conditions.includes(cond);
                    return (
                      <td key={m.model_key}>
                        {d ? <VlmBadge label={d} /> : <span className={`state ${usable ? "parse_error" : "not_run"}`}>{usable ? "parse error" : "not run"}</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr>
                <td colSpan={3} className="small muted">In this sample</td>
                {models.map((m) => (
                  <td key={m.model_key} className="tiny muted">
                    R {tally[m.model_key].Reliable} · U {tally[m.model_key].Uncertain} · Ur {tally[m.model_key].Unreliable}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
