import { useCallback, useState } from "react";
import { api, IS_STATIC } from "./api/client";
import { snapshotMeta, type SnapshotMeta } from "./api/static";
import { num } from "./utils/format";
import { TopNav } from "./components/navigation/TopNav";
import { useApi } from "./hooks/useApi";
import { ContextShift } from "./sections/ContextShift";
import { Difficulty } from "./sections/Difficulty";
import { Explorer } from "./sections/Explorer";
import { HumanReview } from "./sections/HumanReview";
import { Method } from "./sections/Method";
import { ModelScale } from "./sections/ModelScale";
import { Overview } from "./sections/Overview";
import { SemanticGT } from "./sections/SemanticGT";
import type { ModelInfo, ReviewResults, Summary } from "./types/api";

function initialDetection(): string | null {
  const p = new URLSearchParams(window.location.search).get("d");
  return p && /^sample_\d{6}$/.test(p) ? p : null;
}

export default function App() {
  const summary = useApi<Summary>(() => api.getSummary(), []);
  const models = useApi<ModelInfo[]>(() => api.getModels(), []);
  const [selected, setSelected] = useState<string | null>(initialDetection);
  const [review, setReview] = useState<ReviewResults | null>(null);
  const meta = useApi<SnapshotMeta | null>(() => (IS_STATIC ? snapshotMeta() : Promise.resolve(null)), []);

  const openDetection = useCallback((id: string) => {
    setSelected(id);
    document.getElementById("explorer")?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const backendDown = summary.error && models.error;
  const m = models.data ?? [];

  return (
    <>
      <TopNav />
      <main className="page">
        {IS_STATIC && meta.data && (
          <p className="note small" style={{ marginTop: 24 }}>
            Public snapshot ({meta.data.generated.slice(0, 10)}). All aggregate results cover the full study; the Detection Explorer
            includes {num(meta.data.n_detections_included)} of the {num(meta.data.n_detections_total)} detections (every curated case,
            the review sample and a random fill) and images are re-encoded ({meta.data.image_encoding}). Blind-review answers stay in
            your browser.
          </p>
        )}
        {backendDown && !IS_STATIC && (
          <p className="error" style={{ marginTop: 24 }}>
            The demo API is not reachable ({summary.error}). Start it with{" "}
            <span className="mono">cd demo/backend && python3 -m uvicorn app.main:app --port 8000</span>.
          </p>
        )}
        <Overview summary={summary.data} models={m} />
        <Explorer models={m} selected={selected} setSelected={setSelected} />
        <ContextShift models={m} />
        <ModelScale />
        <Difficulty />
        <HumanReview results={review} onResults={setReview} onOpenDetection={openDetection} />
        <SemanticGT visitor={review} />
        <Method summary={summary.data} />
      </main>
      <footer>
        <div className="page">
          WILD PALM research demo. Read-only view of stored research artifacts (outputs/, paper/analysis/); no model inference
          is run.{" "}
          {IS_STATIC ? "Static snapshot; visitor review answers never leave your browser." : "Visitor review answers are written only under demo/data/."}
          {summary.data && (
            <span className="mono"> · {summary.data.sources.decisions}</span>
          )}
        </div>
      </footer>
    </>
  );
}
