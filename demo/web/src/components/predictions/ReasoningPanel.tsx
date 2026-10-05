import { api } from "../../api/client";
import { useApi } from "../../hooks/useApi";
import type { Condition, Reasoning } from "../../types/api";
import { VlmBadge } from "../common/Labels";

export function ReasoningPanel({
  detectionId,
  model,
  modelName,
  condition,
}: {
  detectionId: string;
  model: string;
  modelName: string;
  condition: Condition;
}) {
  const r = useApi<Reasoning>(() => api.getReasoning(detectionId, model, condition), [detectionId, model, condition]);
  return (
    <div className="reasoning">
      <div>
        <div className="field-label">Stored response</div>
        <div style={{ fontWeight: 600 }}>{modelName}</div>
        <div className="mono small muted">{condition}</div>
        {r.data?.decision && (
          <div style={{ marginTop: 8 }}>
            <VlmBadge label={r.data.decision} />
          </div>
        )}
        {r.data?.runtime_seconds !== undefined && <div className="tiny muted" style={{ marginTop: 6 }}>{r.data.runtime_seconds?.toFixed(1)} s generation</div>}
      </div>
      <div>
        {r.loading && <div className="loading">reading prediction file…</div>}
        {r.error && <div className="error">{r.error}</div>}
        {r.data && !r.data.available && <p className="muted">{r.data.reason}</p>}
        {r.data?.available && (
          <>
            {r.data.parse_error ? (
              <>
                <p className="error">Parse error: {r.data.parse_error}</p>
                <p className="small muted">Raw model output (not repaired, not re-parsed):</p>
                <pre className="prompt-block" style={{ maxHeight: 240, overflow: "auto" }}>
                  {r.data.raw_response}
                  {r.data.raw_response_truncated ? "\n…" : ""}
                </pre>
              </>
            ) : (
              <div className="quote">{r.data.visual_reasoning || <span className="muted">No visual reasoning text was returned.</span>}</div>
            )}
            {r.data.confidence_reasoning && <p className="small muted" style={{ marginTop: 10 }}>Confidence reasoning: {r.data.confidence_reasoning}</p>}
            {r.data.consistent_with_analysis_table === false && (
              <p className="error small">The decision in this file differs from the analysis table; the table value is shown in the matrix.</p>
            )}
            <div className="source">{r.data.source}</div>
          </>
        )}
      </div>
    </div>
  );
}
