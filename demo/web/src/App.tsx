import { useCallback, useState } from "react";
import { api } from "./api/client";
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
        {backendDown && (
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
          is run. Visitor review answers are written only under demo/data/.
          {summary.data && (
            <span className="mono"> · {summary.data.sources.decisions}</span>
          )}
        </div>
      </footer>
    </>
  );
}
