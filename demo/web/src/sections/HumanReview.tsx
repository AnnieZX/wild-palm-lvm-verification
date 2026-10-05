import { useState } from "react";
import { api, IS_STATIC } from "../api/client";
import { ErrorNote, Section } from "../components/common/Section";
import { Segmented } from "../components/common/Segmented";
import { ReviewReveal } from "../components/review/ReviewReveal";
import { ReviewTask } from "../components/review/ReviewTask";
import { useApi } from "../hooks/useApi";
import type { ReviewDecks, ReviewResults, ReviewSession } from "../types/api";

interface Props {
  results: ReviewResults | null;
  onResults: (r: ReviewResults | null) => void;
  onOpenDetection: (id: string) => void;
}

export function HumanReview({ results, onResults, onOpenDetection }: Props) {
  const decks = useApi<ReviewDecks>(() => api.getReviewDecks(), []);
  const [deck, setDeck] = useState("unmatched");
  const [count, setCount] = useState(10);
  const [session, setSession] = useState<ReviewSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  const start = async () => {
    setError(null);
    onResults(null);
    try {
      setSession(await api.createReviewSession(count, deck));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const reveal = async () => {
    if (!session) return;
    try {
      onResults(await api.getReviewResults(session.session_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const restart = () => {
    setSession(null);
    onResults(null);
  };

  return (
    <Section
      id="review"
      kicker="06 · Try Human Review"
      title="Is the detected region actually a palm?"
      lede={
        <>
          Repeat the project’s blind semantic audit on a random sample of real detections. You see only what the reviewer saw:
          the source patch and the YOLO box. No LabelMe annotation, YOLO confidence, IoU or model output is shown until you
          finish.{" "}
          {IS_STATIC
            ? "Your answers stay in your browser and never touch the official annotations. This snapshot draws from a fixed random subset of each deck."
            : "Your answers are stored only in the demo’s own folder and never touch the official annotations."}
        </>
      }
    >
      {error && <ErrorNote error={error} />}
      {!session && (
        <div className="cols-2" style={{ alignItems: "end" }}>
          <div className="stack">
            <div>
              <span className="field-label">Sample from</span>
              {decks.data ? (
                <div className="stack">
                  {Object.entries(decks.data.decks).map(([k, d]) => (
                    <label key={k} style={{ display: "grid", gridTemplateColumns: "20px 1fr", gap: 8, cursor: "pointer" }}>
                      <input type="radio" name="deck" checked={deck === k} onChange={() => setDeck(k)} />
                      <span>
                        <strong>{d.title}</strong>
                        {k === "unmatched" && <span className="muted"> (recommended)</span>}
                        <br />
                        <span className="small muted">{d.description}</span>
                      </span>
                    </label>
                  ))}
                </div>
              ) : (
                <div className="loading">loading…</div>
              )}
            </div>
            <div className="row">
              <span className="field-label" style={{ margin: 0 }}>Items</span>
              <Segmented value={String(count)} onChange={(v) => setCount(Number(v))} options={["5", "10", "20"].map((v) => ({ value: v, label: v }))} />
              <button className="btn primary" onClick={start} disabled={!decks.data}>
                Start blind review
              </button>
            </div>
          </div>
          <div className="note small">
            Keyboard: <span className="mono">P</span> palm, <span className="mono">N</span> non-palm, <span className="mono">A</span> ambiguous,{" "}
            <span className="mono">S</span> skip, <span className="mono">← →</span> move. Wording, labels and the zoomed crop follow the
            project’s review tool (<span className="mono">src/semantic_review/</span>).
          </div>
        </div>
      )}
      {session && !results && <ReviewTask session={session} onUpdate={setSession} onReveal={reveal} />}
      {session && results && (
        <>
          <ReviewReveal results={results} onOpenDetection={onOpenDetection} />
          <div className="row" style={{ marginTop: 32 }}>
            <button className="btn" onClick={restart}>Review another sample</button>
            <a className="btn primary" href="#semantic" style={{ textDecoration: "none" }}>See the full audit →</a>
          </div>
        </>
      )}
    </Section>
  );
}
