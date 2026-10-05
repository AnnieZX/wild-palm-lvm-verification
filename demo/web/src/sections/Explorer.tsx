import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ErrorNote, Loading, Section } from "../components/common/Section";
import { SemanticBadge } from "../components/common/Labels";
import { ConditionBar, ConditionImage, ConditionPrompt } from "../components/detection/ConditionView";
import { DetectionBrowser } from "../components/detection/DetectionBrowser";
import { PatchView } from "../components/detection/PatchView";
import { PredictionMatrix } from "../components/predictions/PredictionMatrix";
import { ReasoningPanel } from "../components/predictions/ReasoningPanel";
import { useApi } from "../hooks/useApi";
import type { Condition, DetectionDetail, ModelInfo, Predictions } from "../types/api";
import { CONDITION_DEFS, UNMATCHED_REASON_TEXT } from "../utils/research";

interface Props {
  models: ModelInfo[];
  selected: string | null;
  setSelected: (id: string) => void;
}

export function Explorer({ models, selected, setSelected }: Props) {
  const [condition, setCondition] = useState<Condition>("A1");
  const [picked, setPicked] = useState<{ model: string; condition: Condition } | null>(null);
  const [showLabelme, setShowLabelme] = useState(true);
  const [showOthers, setShowOthers] = useState(true);

  useEffect(() => {
    if (selected) return;
    api
      .getCurated()
      .then((cases) => {
        const c = cases.find((x) => x.category === "strong_A2_to_A5_change") ?? cases[0];
        setSelected(c ? c.sample_id : "sample_000001");
      })
      .catch(() => setSelected("sample_000001"));
  }, [selected]);

  useEffect(() => {
    if (!selected) return;
    const url = new URL(window.location.href);
    url.searchParams.set("d", selected);
    window.history.replaceState(null, "", url);
    setPicked(null);
  }, [selected]);

  const det = useApi<DetectionDetail | null>(() => (selected ? api.getDetection(selected) : Promise.resolve(null)), [selected]);
  const preds = useApi<Predictions | null>(() => (selected ? api.getPredictions(selected) : Promise.resolve(null)), [selected]);

  const pick = (model: string, c: Condition) => {
    setPicked({ model, condition: c });
    setCondition(c);
  };

  return (
    <Section
      id="explorer"
      kicker="02 · Detection Explorer"
      title="One detection, five views, many verifiers"
      lede={
        <>
          Pick any of the 5,747 YOLO detections. The box stays fixed; switch A1–A5 to change what the VLM is shown, and read
          what every checkpoint actually answered. Nothing on this page is re-inferred: images, prompts and decisions are the
          stored research artifacts.
        </>
      }
    >
      <div className="explorer">
        <DetectionBrowser selected={selected} onSelect={setSelected} models={models} />
        <div>
          {det.error && <ErrorNote error={det.error} />}
          {!det.data && !det.error && <Loading what="loading detection" />}
          {det.data && (
            <>
              <DetectionHeader d={det.data} />
              <ConditionBar value={condition} onChange={setCondition} />
              <div className="viewers">
                <div>
                  <div className="viewer-caption">
                    <span className="title">Source patch · YOLO detection</span>
                    <span className="mono tiny muted">{det.data.image_id}.png</span>
                  </div>
                  <div className="imgframe">
                    <PatchView
                      imageUrl={det.data.raw_image_url}
                      width={det.data.image_width}
                      height={det.data.image_height}
                      target={det.data.bbox_xywh}
                      others={showOthers ? det.data.other_detections.map((o) => o.bbox_xywh) : []}
                      labelme={showLabelme ? det.data.labelme.palm_boxes_xywh : []}
                    />
                  </div>
                  <div className="legend">
                    <span>
                      <i className="swatch" style={{ borderColor: "#e0b800" }} /> this detection
                    </span>
                    <label>
                      <input type="checkbox" checked={showOthers} onChange={(e) => setShowOthers(e.target.checked)} />
                      <i className="swatch" style={{ borderColor: "#999" }} /> other YOLO detections ({det.data.other_detections.length})
                    </label>
                    <label>
                      <input type="checkbox" checked={showLabelme} onChange={(e) => setShowLabelme(e.target.checked)} />
                      <i className="swatch dashed" style={{ borderColor: "#2aa7d6" }} /> LabelMe palm boxes ({det.data.labelme.palm_boxes_xywh.length})
                    </label>
                  </div>
                </div>
                <div>
                  <div className="viewer-caption">
                    <span className="title">
                      What the VLM receives · {condition} {CONDITION_DEFS[condition].name.toLowerCase()}
                    </span>
                    {CONDITION_DEFS[condition].imageSameAs && (
                      <span className="tiny muted">same image as {CONDITION_DEFS[condition].imageSameAs}; only the text changes</span>
                    )}
                  </div>
                  <ConditionImage detection={det.data} condition={condition} />
                </div>
              </div>
              <ConditionPrompt detectionId={det.data.detection_id} condition={condition} />
            </>
          )}

          <div className="matrix-wrap">
            <div className="row" style={{ alignItems: "baseline", marginBottom: 6 }}>
              <h3 style={{ fontSize: 22 }}>Checkpoint × condition</h3>
              <span className="small muted">Click a column to switch condition; click a cell to read the stored response.</span>
            </div>
            {preds.error && <ErrorNote error={preds.error} />}
            {preds.data && (
              <PredictionMatrix
                predictions={preds.data}
                condition={condition}
                onCondition={setCondition}
                picked={picked}
                onPick={pick}
              />
            )}
            {picked && selected && (
              <ReasoningPanel
                detectionId={selected}
                model={picked.model}
                modelName={models.find((m) => m.model_key === picked.model)?.display ?? picked.model}
                condition={picked.condition}
              />
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}

function DetectionHeader({ d }: { d: DetectionDetail }) {
  const [, , w, h] = d.bbox_xywh;
  return (
    <>
      <div className="det-header">
        <h3>{d.detection_id}</h3>
        <span className="small muted">
          patch {d.image_id} · {d.detections_in_image} YOLO detection{d.detections_in_image === 1 ? "" : "s"} in this patch
        </span>
      </div>
      <div className="metastrip">
        <div>
          <div className="k">YOLO confidence</div>
          <div className="v">
            {d.yolo_confidence.toFixed(3)} <span className="muted small">{d.confidence_quintile}</span>
          </div>
        </div>
        <div>
          <div className="k">Box</div>
          <div className="v">
            {Math.round(w)}×{Math.round(h)} px <span className="muted small">size {d.area_quintile}</span>
          </div>
        </div>
        <div>
          <div className="k">Max IoU with LabelMe</div>
          <div className="v">{d.max_iou === null ? "—" : d.max_iou.toFixed(3)}</div>
        </div>
        <div>
          <div className="k">LabelMe geometry</div>
          <div className="v">
            {d.labelme_matched === null ? "—" : d.labelme_matched ? "matched" : "unmatched"}
          </div>
        </div>
        <div>
          <div className="k">Human semantic review</div>
          <div className="v">
            <SemanticBadge label={d.semantic_label} />
          </div>
        </div>
        <div>
          <div className="k">Why unmatched</div>
          <div className="v small" style={{ maxWidth: 230 }}>
            {d.unmatched_reason ? UNMATCHED_REASON_TEXT[d.unmatched_reason] ?? d.unmatched_reason : <span className="muted">—</span>}
          </div>
        </div>
      </div>
      {!d.semantic_label && d.labelme_matched && (
        <p className="tiny muted" style={{ marginTop: -14, marginBottom: 18 }}>
          Only the 638 LabelMe-unmatched detections were semantically reviewed; matched detections carry no human label.
        </p>
      )}
    </>
  );
}
