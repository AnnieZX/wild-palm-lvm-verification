import { Section } from "../components/common/Section";
import type { Summary } from "../types/api";
import { num } from "../utils/format";
import { CONDITION_DEFS, CONDITIONS } from "../utils/research";

export function Method({ summary }: { summary: Summary | null }) {
  const s = summary;
  return (
    <Section
      id="method"
      kicker="08 · Method"
      title="How the evidence was produced"
      lede="A compact account of the pipeline behind every number on this page. The thesis remains the full reference."
    >
      <div className="pipeline">
        <Step n="01" t="Source patch" d={`912×912 UAV orthomosaic tiles (${num(s?.n_images)} patches in the cohort).`} />
        <Step n="02" t="YOLO detection" d={`YOLO11x proposals with confidence ≥ 0.5: ${num(s?.n_detections)} detections.`} />
        <Step n="03" t="A1–A5 inputs" d="Five fixed renderings of each detection, image and prompt text." />
        <Step n="04" t="VLM verification" d="Each checkpoint answers once per condition, greedy decoding, JSON output." />
        <Step n="05" t="R / U / Ur" d="Reliable, Uncertain or Unreliable; parse failures kept as such." />
        <Step n="06" t="Behaviour analysis" d="Distributions, transitions, bootstrap intervals, label-usage diagnostics." />
      </div>

      <h4 style={{ fontSize: 20, margin: "48px 0 12px" }}>Input conditions (as executed)</h4>
      <table className="data">
        <thead>
          <tr>
            <th>Code</th>
            <th>Condition directory</th>
            <th>Image</th>
            <th>Prompt adds</th>
          </tr>
        </thead>
        <tbody>
          {CONDITIONS.map((c) => (
            <tr key={c}>
              <td className="mono">{c}</td>
              <td className="mono small">{CONDITION_DEFS[c].dir}</td>
              <td className="small">{CONDITION_DEFS[c].image}</td>
              <td className="small">{CONDITION_DEFS[c].text}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="tiny muted" style={{ marginTop: 8 }}>
        A1→A2→A3 change only text. A2→A4/A5 change the image representation together with its description paragraph, so they
        are not image-only contrasts. Source: outputs/verification_ablation_5747/*/prompts and prompt_index.csv.
      </p>

      <div className="cols-2" style={{ marginTop: 48 }}>
        <div>
          <h4 style={{ fontSize: 20, marginBottom: 10 }}>Geometric matching (LabelMe)</h4>
          <p className="small">
            Each detection is compared with the LabelMe palm annotations of its patch (axis-aligned envelope of each annotation’s
            points). Greedy one-to-one matching by descending IoU, threshold 0.5: <strong>{num(s?.labelme_matched)} matched</strong>,{" "}
            <strong>{num(s?.labelme_unmatched)} unmatched</strong>. This measures whether a detection lines up with an existing annotation, nothing more.
          </p>
        </div>
        <div>
          <h4 style={{ fontSize: 20, marginBottom: 10 }}>Semantic validation (human)</h4>
          <p className="small">
            A reviewer looked at each unmatched detection, seeing only the raw patch and the target box, and answered “Does the
            highlighted YOLO detection correspond to a real palm?” with palm, non-palm or ambiguous. This measures whether the
            detected object is a palm, independent of whether anyone annotated it.
          </p>
        </div>
      </div>

      <table className="data" style={{ marginTop: 32 }}>
        <thead>
          <tr>
            <th>Layer</th>
            <th>Question</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Primary</td>
            <td className="small">How do A1–A5 change VLM verification behaviour?</td>
            <td className="small">Stored VLM decisions only; needs no ground truth.</td>
          </tr>
          <tr>
            <td>Secondary</td>
            <td className="small">Do VLM decisions agree with LabelMe alignment?</td>
            <td className="small">Protocol v2 (IoU ≥ 0.5); alignment metrics, not semantic accuracy.</td>
          </tr>
          <tr>
            <td>Audit</td>
            <td className="small">Are the unmatched detections palms?</td>
            <td className="small">Human semantic review of all {num(s?.labelme_unmatched)} unmatched detections.</td>
          </tr>
        </tbody>
      </table>
    </Section>
  );
}

function Step({ n, t, d }: { n: string; t: string; d: string }) {
  return (
    <div>
      <span className="n">{n}</span>
      <h4>{t}</h4>
      <p>{d}</p>
    </div>
  );
}
