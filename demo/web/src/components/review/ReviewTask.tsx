import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { ReviewSession, SemanticLabel, VisitorLabel } from "../../types/api";
import { reviewCropWindow, SEMANTIC_NAME } from "../../utils/research";
import { PatchView } from "../detection/PatchView";

const KEYS: Record<string, VisitorLabel> = { p: "palm", n: "non_palm", a: "ambiguous", s: "skip" };
const BUTTONS: { label: VisitorLabel; key: string; text: string }[] = [
  { label: "palm", key: "P", text: "Palm" },
  { label: "non_palm", key: "N", text: "Non-palm" },
  { label: "ambiguous", key: "A", text: "Ambiguous" },
  { label: "skip", key: "S", text: "Skip" },
];

/**
 * Blind review of one session. Shows only what the project's reviewer saw: the raw patch and the
 * target box (context and zoomed crop). No ID, confidence, IoU, LabelMe box or model output.
 */
export function ReviewTask({ session, onUpdate, onReveal }: {
  session: ReviewSession;
  onUpdate: (s: ReviewSession) => void;
  onReveal: () => void;
}) {
  const [index, setIndex] = useState(() => {
    const first = session.items.findIndex((i) => !i.visitor_label);
    return first === -1 ? 0 : first;
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const item = session.items[index];
  const n = session.items.length;

  const answer = useCallback(
    async (label: VisitorLabel) => {
      if (busy) return;
      setBusy(true);
      setErr(null);
      try {
        const next = await api.submitReview(session.session_id, index, label);
        onUpdate(next);
        const after = next.items.findIndex((it, i) => i > index && !it.visitor_label);
        const anyOpen = next.items.findIndex((it) => !it.visitor_label);
        if (after !== -1) setIndex(after);
        else if (anyOpen !== -1) setIndex(anyOpen);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, index, session.session_id, onUpdate],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      const k = e.key.toLowerCase();
      if (KEYS[k]) {
        e.preventDefault();
        answer(KEYS[k]);
      } else if (e.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
      else if (e.key === "ArrowRight") setIndex((i) => Math.min(n - 1, i + 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [answer, n]);

  const w = item.image_width ?? 912;
  const h = item.image_height ?? 912;
  const crop = reviewCropWindow(item.bbox_xywh, w, h);

  return (
    <div>
      <div className="row" style={{ marginBottom: 14 }}>
        <span className="mono small">
          Detection {index + 1} / {n}
        </span>
        <div className="progress">
          {session.items.map((it, i) => (
            <span key={i} className={i === index ? "cur" : it.visitor_label ? "done" : ""} onClick={() => setIndex(i)} title={`item ${i + 1}`} />
          ))}
        </div>
        <span className="spacer" />
        <span className="tiny muted">{session.deck_info.title}</span>
      </div>
      <div className="review-stage">
        <div>
          <div className="viewer-caption"><span className="title">Context · full patch</span></div>
          <div className="imgframe">
            <PatchView imageUrl={item.image_url} width={w} height={h} target={item.bbox_xywh} label="review context" />
          </div>
        </div>
        <div>
          <div className="viewer-caption"><span className="title">Zoom · target region</span></div>
          <div className="imgframe">
            <PatchView imageUrl={item.image_url} width={w} height={h} target={item.bbox_xywh} window={crop} label="review crop" />
          </div>
        </div>
      </div>
      <div className="review-q">{session.question}</div>
      <div className="review-actions">
        {BUTTONS.map((b) => (
          <button
            key={b.label}
            className={`btn ${b.label}${item.visitor_label === b.label ? " picked" : ""}`}
            onClick={() => answer(b.label)}
            disabled={busy}
          >
            <kbd>{b.key}</kbd>
            {b.text}
          </button>
        ))}
        <span className="spacer" />
        <button className="btn" onClick={() => setIndex(Math.max(0, index - 1))} disabled={index === 0}>
          ← Previous
        </button>
        <button className="btn" onClick={() => setIndex(Math.min(n - 1, index + 1))} disabled={index === n - 1}>
          Next →
        </button>
      </div>
      {err && <p className="error">{err}</p>}
      <details style={{ marginTop: 18 }}>
        <summary>Label definitions (project review protocol)</summary>
        <div style={{ marginTop: 10, maxWidth: "var(--measure)" }}>
          {(Object.keys(session.definitions) as SemanticLabel[]).map((k) => (
            <p key={k} className="small">
              <strong>{SEMANTIC_NAME[k]}</strong>: {session.definitions[k]}
            </p>
          ))}
          <p className="small muted">Skip moves on without a label. You can change an answer until you reveal the results.</p>
        </div>
      </details>
      <div className="row" style={{ marginTop: 26, borderTop: "1px solid var(--rule)", paddingTop: 18 }}>
        <span className="small muted">
          {session.n_answered} of {n} answered. Research labels and model decisions stay hidden until every item is answered or skipped.
        </span>
        <span className="spacer" />
        <button className="btn primary" disabled={!session.complete} onClick={onReveal}>
          Reveal the audit
        </button>
      </div>
    </div>
  );
}
