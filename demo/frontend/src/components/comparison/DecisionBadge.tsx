import type { VerificationDecision } from "@shared/types";

import { decisionBadgeDark } from "@/theme/decision-styles";

export interface DecisionBadgeProps {
  decision: VerificationDecision;
  size?: "sm" | "md";
  className?: string;
}

export function DecisionBadge({ decision, size = "md", className = "" }: DecisionBadgeProps) {
  const sizeClass = size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-sm";

  return (
    <span
      className={`inline-flex rounded-md font-semibold ring-1 ring-inset ${sizeClass} ${decisionBadgeDark[decision]} ${className}`}
    >
      {decision}
    </span>
  );
}
