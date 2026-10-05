import type { ModelInfo, Summary } from "../types/api";
import { num } from "../utils/format";

export function Overview({ summary, models }: { summary: Summary | null; models: ModelInfo[] }) {
  const complete = models.filter((m) => m.n_usable_conditions === 5);
  const partialRuns = models.filter((m) => m.n_usable_conditions > 0 && m.n_usable_conditions < 5);
  const pending = models.filter((m) => m.n_usable_conditions === 0);
  const sem = summary?.semantic_review ?? {};
  return (
    <header id="overview" className="hero">
      <div className="hero-kicker">WILD PALM · INTERACTIVE RESEARCH DEMO</div>
      <h1>
        WILD PALM
        <span className="sub">Visual context and VLM verification of palm detections</span>
      </h1>
      <div className="hero-grid">
        <div>
          <ol className="steps">
            <li>A YOLO detector proposes a palm in a UAV orthomosaic patch.</li>
            <li>A vision-language model is asked whether that detection is Reliable, Uncertain or Unreliable.</li>
            <li>We systematically change what visual context the model can see (A1–A5) while the detection stays fixed.</li>
            <li>We study how decisions move, and keep geometric agreement with LabelMe separate from human semantic validity.</li>
          </ol>
        </div>
        <div className="facts">
          <Fact value={summary ? num(summary.n_detections) : "…"} label={`YOLO detections (confidence ≥ ${summary?.yolo_confidence_min.toFixed(1) ?? "0.5"})`} />
          <Fact value={summary ? num(summary.n_images) : "…"} label="source patches" />
          <Fact value={summary ? String(summary.conditions.length) : "…"} label="input conditions per detection" />
          <Fact
            value={String(complete.length)}
            label={`checkpoints with all five conditions${partialRuns.length ? `; ${partialRuns.length} more with some` : ""}${pending.length ? `; ${pending.length} pending` : ""}`}
          />
          <Fact value={summary ? num(summary.n_decisions) : "…"} label="stored VLM decisions analysed" />
          <Fact
            value={summary ? num(summary.labelme_unmatched) : "…"}
            label={`IoU-unmatched detections, all human-reviewed: ${sem.palm ?? 0} palm · ${sem.ambiguous ?? 0} ambiguous · ${sem.non_palm ?? 0} non-palm`}
          />
        </div>
      </div>

      <div className="findings">
        <Finding href="#explorer" tag="A" title="Context matters" body="Removing or changing visual context can substantially change a VLM’s verification decision for the same detection." />
        <Finding href="#context-shift" tag="B" title="Response is checkpoint-specific" body="Some checkpoints move toward Uncertain, others toward Unreliable, others barely move." />
        <Finding href="#scale" tag="C" title="Scale changes behaviour" body="Within a family, checkpoints of different size respond differently; small ones can collapse to one label. No general scaling law." />
        <Finding href="#difficulty" tag="D" title="Detection characteristics matter" body="Smaller and lower-confidence detections are often more context-sensitive, with checkpoint-specific exceptions." />
        <Finding href="#semantic" tag="E" title="LabelMe is non-exhaustive" body="IoU-unmatched does not mean non-palm. Geometric agreement and semantic validity are different constructs." />
      </div>
    </header>
  );
}

function Fact({ value, label }: { value: string; label: string }) {
  return (
    <div className="fact">
      <div className="fact-value">{value}</div>
      <div className="fact-label">{label}</div>
    </div>
  );
}

function Finding({ href, tag, title, body }: { href: string; tag: string; title: string; body: string }) {
  return (
    <a className="finding" href={href}>
      <span className="tag">FINDING {tag}</span>
      <h4>{title}</h4>
      <p>{body}</p>
    </a>
  );
}
