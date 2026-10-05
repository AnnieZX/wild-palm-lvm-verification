import { useState } from "react";
import { api } from "../../api/client";
import { useApi } from "../../hooks/useApi";
import type { Condition, DetectionDetail, PromptInfo } from "../../types/api";
import { CONDITION_DEFS, CONDITIONS } from "../../utils/research";

export function ConditionBar({ value, onChange }: { value: Condition; onChange: (c: Condition) => void }) {
  return (
    <div className="condbar" role="tablist" aria-label="Input condition">
      {CONDITIONS.map((c) => (
        <button key={c} role="tab" aria-selected={c === value} className={c === value ? "on" : ""} onClick={() => onChange(c)}>
          <span className="code">{c}</span>
          <span className="cname">{CONDITION_DEFS[c].name}</span>
        </button>
      ))}
    </div>
  );
}

/** The exact image file the VLMs received for this condition (from the condition's prompt_index.csv). */
export function ConditionImage({ detection, condition }: { detection: DetectionDetail; condition: Condition }) {
  const url = detection.condition_inputs[condition].image_url;
  const [loaded, setLoaded] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const wide = condition === "A4";
  return (
    <div className="imgframe" style={{ aspectRatio: wide ? "2 / 1" : "1 / 1" }}>
      <img
        key={url}
        src={url}
        alt={`${condition} model input for ${detection.detection_id}`}
        onLoad={() => setLoaded(url)}
        onError={() => setFailed(url)}
        style={{ opacity: loaded === url ? 1 : 0, transition: "opacity 160ms" }}
      />
      {loaded !== url && failed !== url && <div className="placeholder">loading {condition} input…</div>}
      {failed === url && <div className="placeholder">input image unavailable</div>}
    </div>
  );
}

/** Condition-specific prompt text, as executed; sections that differ from A1 are marked. */
export function ConditionPrompt({ detectionId, condition }: { detectionId: string; condition: Condition }) {
  const cur = useApi<PromptInfo>(() => api.getPrompt(detectionId, condition), [detectionId, condition]);
  const base = useApi<PromptInfo>(() => api.getPrompt(detectionId, "A1"), [detectionId]);
  const def = CONDITION_DEFS[condition];
  if (cur.error) return <div className="error">{cur.error}</div>;
  const p = cur.data;
  const b = base.data;
  return (
    <div className="condinfo">
      <div>
        <div className="k">Image the model receives</div>
        <p>{def.image}</p>
        <div className="k" style={{ marginTop: 14 }}>
          Prompt text
        </div>
        <p>{def.text}</p>
        <p className="small muted">
          Everything else in the prompt (task, palm characteristics, decision definitions, output format) is identical across
          A1–A5.
        </p>
      </div>
      <div>
        {p ? (
          <>
            <div className="k">“Input image” paragraph</div>
            <div className={`prompt-block${b && p.input_image_section !== b.input_image_section ? " changed" : ""}`}>
              {(p.input_image_section ?? "").replace(/^Input image\n?/, "")}
            </div>
            <div className="k" style={{ marginTop: 12 }}>
              Metadata block
            </div>
            <div className={`prompt-block${b && p.metadata_section !== b.metadata_section ? " changed" : ""}`}>{p.metadata_section}</div>
            <details style={{ marginTop: 10 }}>
              <summary>Full executed prompt</summary>
              <pre className="prompt-block">{p.full_text}</pre>
              <div className="source">{p.source}</div>
            </details>
          </>
        ) : (
          <div className="loading">loading prompt…</div>
        )}
      </div>
    </div>
  );
}
