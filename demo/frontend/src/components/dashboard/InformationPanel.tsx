import type { SampleDetail } from "@shared/types";
import type { ModelInfo } from "@shared/types";

import { DecisionBadge } from "@/components/comparison/DecisionBadge";
import { ABLATION_CONDITION_BY_CODE } from "@/lib/constants";
import type { AblationCode } from "@shared/types";

interface InformationPanelProps {
  sample: SampleDetail;
  model: ModelInfo;
  experimentId: string;
  ablation: AblationCode;
}

function formatConfidence(value: number | null): string {
  if (value === null) return "—";
  return value.toFixed(3);
}

function formatIou(value: number): string {
  return value.toFixed(3);
}

function gtDisplay(sample: SampleDetail): string {
  if (sample.matched_gt) return "Positive";
  if (sample.gt_label === "negative") return "Negative";
  return sample.gt_label;
}

export function InformationPanel({
  sample,
  model,
  experimentId,
  ablation,
}: InformationPanelProps) {
  const condition = ABLATION_CONDITION_BY_CODE[ablation] ?? ablation;
  const modelOutput = [
    sample.visual_reasoning.trim(),
    sample.confidence_reasoning.trim(),
  ]
    .filter(Boolean)
    .join("\n\n");

  return (
    <aside
      className="flex h-full flex-col overflow-hidden border-t border-slate-200 bg-white xl:border-l xl:border-t-0"
      aria-label="Sample inspection"
    >
      <div className="border-b border-slate-200 px-3 py-2.5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="wp-label">Inspect</p>
            <p className="truncate font-mono text-sm font-medium text-slate-900">
              {sample.sample_id}
            </p>
          </div>
          {sample.decision ? (
            <DecisionBadge decision={sample.decision} size="sm" />
          ) : (
            <span className="text-xs text-slate-400">No decision</span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        <section aria-labelledby="detect-heading">
          <h3 id="detect-heading" className="wp-label">
            Detection
          </h3>
          <div className="mt-1.5 space-y-0.5">
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">YOLO confidence</span>
              <span className="wp-inspector-value">
                {formatConfidence(sample.yolo_confidence)}
              </span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">Ground truth</span>
              <span className="wp-inspector-value">{gtDisplay(sample)}</span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">Matched GT</span>
              <span className="wp-inspector-value">
                {sample.matched_gt ? "Yes" : "No"}
              </span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">IoU</span>
              <span className="wp-inspector-value">{formatIou(sample.max_iou)}</span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">Image</span>
              <span className="wp-inspector-value truncate" title={sample.image_name}>
                {sample.image_name || "—"}
              </span>
            </div>
          </div>
        </section>

        <section className="wp-section-rule" aria-labelledby="verify-heading">
          <h3 id="verify-heading" className="wp-label">
            Verification
          </h3>
          <div className="mt-1.5 space-y-0.5">
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">Model</span>
              <span className="text-right text-sm text-slate-900">{model.display_name}</span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">Decision</span>
              <span className="wp-inspector-value">{sample.decision ?? "—"}</span>
            </div>
          </div>
        </section>

        <section className="wp-section-rule" aria-labelledby="output-heading">
          <h3 id="output-heading" className="wp-label">
            Model output
          </h3>
          {modelOutput ? (
            <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded border border-slate-200 bg-slate-50 p-2 font-mono text-[11px] leading-relaxed text-slate-700">
              {modelOutput}
            </pre>
          ) : (
            <p className="mt-2 text-xs text-slate-500">No reasoning fields in artifact.</p>
          )}
        </section>

        <section className="wp-section-rule" aria-labelledby="meta-heading">
          <h3 id="meta-heading" className="wp-label">
            Metadata
          </h3>
          <div className="mt-1.5 space-y-0.5">
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">sample_id</span>
              <span className="wp-inspector-value">{sample.sample_id}</span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">experiment</span>
              <span className="wp-inspector-value">{experimentId}</span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">ablation</span>
              <span className="wp-inspector-value">{condition}</span>
            </div>
            <div className="wp-inspector-row">
              <span className="wp-inspector-key">model_key</span>
              <span className="wp-inspector-value">{model.model_key}</span>
            </div>
          </div>
        </section>
      </div>
    </aside>
  );
}
