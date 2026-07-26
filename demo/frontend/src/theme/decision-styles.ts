import type { VerificationDecision } from "@shared/types";

/** Light-panel decision badges (info panel, statistics). */
export const decisionBadgeLight: Record<VerificationDecision, string> = {
  Reliable: "bg-forest-100 text-forest-800 ring-forest-200",
  Uncertain: "bg-warning-100 text-warning-800 ring-warning-200",
  Unreliable: "bg-error-100 text-error-800 ring-error-200",
};

/** Dark-panel decision badges (model comparison strip). */
export const decisionBadgeDark: Record<VerificationDecision, string> = {
  Reliable: "bg-forest-500/15 text-forest-300 ring-forest-500/40",
  Uncertain: "bg-warning-500/15 text-warning-300 ring-warning-500/40",
  Unreliable: "bg-error-500/15 text-error-300 ring-error-500/40",
};

/** Chart segment fills for decision distribution. */
export const decisionChartSegment: Record<VerificationDecision, string> = {
  Reliable: "bg-forest-600",
  Uncertain: "bg-warning-500",
  Unreliable: "bg-error-500",
};
