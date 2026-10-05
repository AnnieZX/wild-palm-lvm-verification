import { useMemo, useState } from "react";
import { api } from "../api/client";
import { LabelKey, StackBar } from "../components/charts/Bars";
import { ErrorNote, Loading, Section, Sources } from "../components/common/Section";
import { Segmented } from "../components/common/Segmented";
import { useApi } from "../hooks/useApi";
import type { Condition, ReviewResults, SemanticReview } from "../types/api";
import { pct } from "../utils/format";
import { CONDITIONS, UNMATCHED_REASON_TEXT } from "../utils/research";

export function SemanticGT({ visitor }: { visitor: ReviewResults | null }) {
  const data = useApi<SemanticReview>(() => api.getSemanticReview(), []);
  const d = data.data;
  const counts = d?.unmatched_review.label_counts;
  const n = d?.unmatched_review.n ?? 0;

  return (
    <Section
      id="semantic"
      kicker="07 · Semantic GT reality check"
      title="IoU-unmatched is not the same as non-palm"
      lede={
        <>
          Protocol v2 labels a detection GT− when it does not match a LabelMe palm box at IoU ≥ 0.5. That reading treats
          LabelMe as exhaustive. A human reviewer looked at every one of those detections.
        </>
      }
    >
      {data.error && <ErrorNote error={data.error} />}
      {data.loading && !d && <Loading />}
      {d && counts && (
        <>
          <div className="equation">
            <div className="term">
              {n.toLocaleString()} IoU-unmatched detections
              <small>
                Cohort detections (YOLO confidence ≥ 0.5) with no LabelMe palm box at IoU ≥ 0.5 under greedy one-to-one matching.
                Historically read as “negatives”.
              </small>
            </div>
            <div className="neq">≠</div>
            <div className="term">
              semantic non-palm
              <small>
                Human review: <strong>{counts.palm}</strong> palm, <strong>{counts.ambiguous}</strong> ambiguous,{" "}
                <strong>{counts.non_palm}</strong> non-palm.
              </small>
            </div>
          </div>

          <div className="cols-2">
            <div>
              <div className="dotgrid" aria-label={`${counts.palm} palm, ${counts.ambiguous} ambiguous, ${counts.non_palm} non-palm`}>
                {Array.from({ length: counts.palm }, (_, i) => <i key={`p${i}`} className="palm" />)}
                {Array.from({ length: counts.ambiguous }, (_, i) => <i key={`a${i}`} className="ambiguous" />)}
                {Array.from({ length: counts.non_palm }, (_, i) => <i key={`n${i}`} className="non_palm" />)}
              </div>
              <div className="bigcount">
                <div style={{ color: "var(--palm)" }}>{counts.palm}<small>palm</small></div>
                <div style={{ color: "var(--amb)" }}>{counts.ambiguous}<small>ambiguous</small></div>
                <div style={{ color: "var(--nonpalm)" }}>{counts.non_palm}<small>non-palm</small></div>
              </div>
              <p className="small muted">
                One dot per detection. Reviewer saw only the raw patch and the target box; review complete for all {n} items
                ({d.unmatched_review.reviewers.join(", ")}, {d.unmatched_review.first_timestamp.slice(0, 10)}).
              </p>
              {visitor && <VisitorLink visitor={visitor} />}
            </div>
            <div>
              <h4 style={{ fontSize: 18, marginBottom: 10 }}>Why were they unmatched?</h4>
              <table className="data">
                <thead>
                  <tr>
                    <th>Geometric reason</th>
                    <th className="num">n</th>
                    <th className="num">palm</th>
                    <th className="num">ambig.</th>
                    <th className="num">non-palm</th>
                  </tr>
                </thead>
                <tbody>
                  {d.unmatched_reason_by_semantic.map((r) => (
                    <tr key={r.unmatched_reason_geometric}>
                      <td className="small">{UNMATCHED_REASON_TEXT[r.unmatched_reason_geometric] ?? r.unmatched_reason_geometric}</td>
                      <td className="num">{r.total}</td>
                      <td className="num">{r.palm}</td>
                      <td className="num">{r.ambiguous}</td>
                      <td className="num">{r.non_palm}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="small" style={{ marginTop: 14 }}>
                Most unmatched palms simply have no LabelMe box, or one that overlaps below the 0.5 IoU threshold. LabelMe
                annotations are non-exhaustive, so an unmatched detection can still be a valid palm.
              </p>
            </div>
          </div>

          <SpecificityPanel d={d} />
          <PalmVerdicts d={d} />
          <Pilot d={d} />

          <div style={{ marginTop: 48 }}>
            <h4 style={{ fontSize: 18, marginBottom: 10 }}>What the audit does and does not support</h4>
            <ul className="small" style={{ maxWidth: "var(--measure)", paddingLeft: 18 }}>
              <li>One reviewer; no inter-rater agreement was measured. Ambiguous cases were kept separate, not forced into a class.</li>
              <li>Only the {n} unmatched detections were reviewed; the {(5747 - n).toLocaleString()} LabelMe-matched detections have no semantic labels.</li>
              <li>The question was “is this a palm?”; box tightness and localisation quality were not scored.</li>
              <li>With 0 confirmed non-palms in the cohort, semantic non-palm specificity cannot be estimated and is not claimed.</li>
            </ul>
          </div>
        </>
      )}
      <Sources paths={d?.sources ?? []} />
    </Section>
  );
}

function VisitorLink({ visitor }: { visitor: ReviewResults }) {
  const palm = visitor.items.filter((i) => i.visitor_label === "palm").length;
  const non = visitor.items.filter((i) => i.visitor_label === "non_palm").length;
  const isUnmatched = visitor.deck === "unmatched";
  return (
    <p className="note" style={{ marginTop: 18 }}>
      In your session you labelled <strong>{palm}</strong> of {visitor.items.length} detections as palm and <strong>{non}</strong> as
      non-palm.{" "}
      {isUnmatched
        ? "Every one of them was IoU-unmatched, i.e. a Protocol-v2 “negative”."
        : "Those came from the lower-confidence pilot, outside the main cohort."}
    </p>
  );
}

function SpecificityPanel({ d }: { d: SemanticReview }) {
  const models = useMemo(() => Array.from(new Set(d.alignment_cells_semantic_composition.map((r) => r.model))), [d]);
  const [model, setModel] = useState(models.includes("Qwen2.5-VL-7B") ? "Qwen2.5-VL-7B" : models[0]);
  const rows = CONDITIONS.map((c) => {
    const tn = d.alignment_cells_semantic_composition.find((r) => r.model === model && r.condition === c && r.alignment_cell.startsWith("TN"));
    const fp = d.alignment_cells_semantic_composition.find((r) => r.model === model && r.condition === c && r.alignment_cell.startsWith("FP"));
    const met = d.alignment_metrics.find((r) => r.model === model && r.condition === c);
    return { c, tn, fp, met };
  }).filter((r) => r.tn || r.fp || r.met);

  return (
    <div style={{ marginTop: 56 }}>
      <h4 style={{ fontSize: 22, marginBottom: 8 }}>What this does to “specificity”</h4>
      <p style={{ maxWidth: "var(--measure)" }}>
        Protocol-v2 specificity is the share of decided IoU-unmatched detections that a VLM called Unreliable. It measures
        agreement with LabelMe geometry. Read against the audit, a “true negative” here is mostly a VLM rejecting a detection
        that a human judged to be a palm.
      </p>
      <div className="row" style={{ margin: "14px 0" }}>
        <span className="field-label" style={{ margin: 0 }}>Checkpoint</span>
        <select className="input" style={{ width: 240 }} value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => <option key={m}>{m}</option>)}
        </select>
      </div>
      <table className="data">
        <thead>
          <tr>
            <th>Condition</th>
            <th className="num">Alignment specificity</th>
            <th className="num">Unmatched → Unreliable</th>
            <th>…of which (human audit)</th>
            <th className="num">Unmatched → Reliable</th>
            <th>…of which (human audit)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ c, tn, fp, met }) => (
            <tr key={c}>
              <td className="mono">{c}</td>
              <td className="num">{pct(met?.align_specificity ?? null)}</td>
              <td className="num">{tn?.n ?? "—"}</td>
              <td className="small">{tn ? `${tn.n_palm} palm · ${tn.n_ambiguous} ambiguous · ${tn.n_non_palm} non-palm` : "—"}</td>
              <td className="num">{fp?.n ?? "—"}</td>
              <td className="small">{fp ? `${fp.n_palm} palm · ${fp.n_ambiguous} ambiguous · ${fp.n_non_palm} non-palm` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="tiny muted" style={{ marginTop: 8 }}>
        Alignment specificity excludes Uncertain. Semantic composition is shown for full-clean cells (identical N = 5,747).
      </p>
    </div>
  );
}

function PalmVerdicts({ d }: { d: SemanticReview }) {
  const [cond, setCond] = useState<Condition>("A1");
  const models = Array.from(new Set(d.vlm_verdicts_by_semantic_label.map((r) => r.model)));
  return (
    <div style={{ marginTop: 56 }}>
      <div className="row" style={{ alignItems: "baseline" }}>
        <h4 style={{ fontSize: 22 }}>How VLMs judged the {d.unmatched_review.label_counts.palm} human-confirmed palms</h4>
        <span className="spacer" />
        <Segmented<Condition> value={cond} onChange={setCond} options={CONDITIONS.map((c) => ({ value: c, label: c }))} />
      </div>
      <p className="small muted" style={{ maxWidth: "var(--measure)", marginTop: 6 }}>
        Palm and Reliable are different statements. These bars show VLM verification labels on IoU-unmatched detections the
        reviewer labelled palm; switch conditions to see the same palms judged with less or more context.
      </p>
      <div style={{ marginTop: 10 }}><LabelKey /></div>
      <table className="data" style={{ marginTop: 8 }}>
        <tbody>
          {models.map((m) => {
            const r = d.vlm_verdicts_by_semantic_label.find((x) => x.model === m && x.condition === cond && x.semantic_label === "palm");
            return (
              <tr key={m}>
                <td style={{ width: 200 }}>{m}</td>
                <td>{r ? <StackBar rates={{ Reliable: r.rate_R, Uncertain: r.rate_U, Unreliable: r.rate_Ur }} /> : <span className="state not_run">not available</span>}</td>
                <td className="num tiny muted" style={{ width: 220 }}>
                  {r ? `R ${pct(r.rate_R, 0)} · U ${pct(r.rate_U, 0)} · Ur ${pct(r.rate_Ur, 0)}` : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Pilot({ d }: { d: SemanticReview }) {
  if (!d.pilot_by_confidence_bin.length) return null;
  const total = d.pilot_by_confidence_bin.reduce((s, r) => ({ palm: s.palm + r.palm, amb: s.amb + r.ambiguous, non: s.non + r.non_palm }), { palm: 0, amb: 0, non: 0 });
  return (
    <div style={{ marginTop: 56 }} className="cols-2">
      <div>
        <h4 style={{ fontSize: 22, marginBottom: 8 }}>Non-palms do exist, below the cohort threshold</h4>
        <p className="small" style={{ maxWidth: "var(--measure)" }}>
          A separate blind pilot drew 100 YOLO detections from each confidence bin between 0.10 and 0.50, outside the 5,747
          cohort and without VLM predictions. It found {total.non} confirmed non-palms. The absence of non-palms is a property
          of the ≥ 0.5 cohort, not of the detector at every threshold.
        </p>
        <p className="note warn small">
          Equal 100-per-bin sampling: the overall shares are not prevalence estimates. Per-bin counts are the reportable quantity.
        </p>
      </div>
      <table className="data">
        <thead>
          <tr>
            <th>YOLO confidence</th>
            <th className="num">palm</th>
            <th className="num">ambiguous</th>
            <th className="num">non-palm</th>
            <th className="num">bin population</th>
          </tr>
        </thead>
        <tbody>
          {d.pilot_by_confidence_bin.map((r) => (
            <tr key={r.confidence_bin}>
              <td className="mono small">{r.confidence_bin}</td>
              <td className="num">{r.palm}</td>
              <td className="num">{r.ambiguous}</td>
              <td className="num">{r.non_palm}</td>
              <td className="num muted">{r.bin_population.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
