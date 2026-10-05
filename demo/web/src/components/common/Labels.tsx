import type { CellState, Degeneracy, SemanticLabel, VisitorLabel, VlmLabel } from "../../types/api";
import {
  CELL_STATE_HELP,
  CELL_STATE_TEXT,
  DEGENERACY_HELP,
  DEGENERACY_TEXT,
  SEMANTIC_NAME,
  VLM_SHORT,
} from "../../utils/research";

/** VLM verification output: filled mark. */
export function VlmBadge({ label, compact = false }: { label: VlmLabel; compact?: boolean }) {
  return (
    <span className={`vlm ${label}${compact ? " compact" : ""}`} title={`VLM verification: ${label}`}>
      {compact ? VLM_SHORT[label] : label}
    </span>
  );
}

/** Human semantic judgement: outlined mark in a separate colour family. */
export function SemanticBadge({ label }: { label: SemanticLabel | VisitorLabel | null | undefined }) {
  if (!label) return <span className="sem none">not reviewed</span>;
  if (label === "skip") return <span className="sem none">skipped</span>;
  return (
    <span className={`sem ${label}`} title={`Human semantic label: ${SEMANTIC_NAME[label]}`}>
      {SEMANTIC_NAME[label]}
    </span>
  );
}

export function CellStateMark({ state }: { state: CellState }) {
  return (
    <span className={`state ${state}`} title={CELL_STATE_HELP[state]}>
      {CELL_STATE_TEXT[state]}
    </span>
  );
}

export function DegeneracyFlag({ value }: { value: Degeneracy }) {
  if (!value) return null;
  return (
    <span className="flag" title={DEGENERACY_HELP[value]}>
      {DEGENERACY_TEXT[value]}
    </span>
  );
}

export function Verdict({ value }: { value: string | null | undefined }) {
  if (!value) return null;
  const cls = value.startsWith("supports")
    ? "supports"
    : value.startsWith("reverses")
      ? "reverses"
      : value.startsWith("null")
        ? "null"
        : "not_evaluable";
  const text = { supports: "Q1 > Q5", reverses: "Q1 < Q5", null: "CI includes 0", not_evaluable: "not evaluable" }[cls];
  return (
    <span className={`verdict ${cls}`} title={value}>
      {text}
    </span>
  );
}
