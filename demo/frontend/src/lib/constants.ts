import type { AblationCode, VerificationDecision } from "@shared/types";

/** Default orthomosaic patch size used before natural image dimensions load. */
export const DEFAULT_IMAGE_SIZE = {
  width: 912,
  height: 912,
} as const;

export const DECISION_OPTIONS: VerificationDecision[] = [
  "Reliable",
  "Uncertain",
  "Unreliable",
];

export interface PromptOption {
  code: AblationCode;
  label: string;
  condition: string;
}

/** Ablation catalog aligned with backend ABLATION_CONDITION_NAMES. */
export const PROMPT_OPTIONS: PromptOption[] = [
  { code: "A1", label: "A1", condition: "A1_overlay_only" },
  { code: "A2", label: "A2", condition: "A2_overlay_confidence" },
  { code: "A3", label: "A3", condition: "A3_overlay_confidence_geometry" },
  { code: "A4", label: "A4", condition: "A4_overlay_crop_confidence" },
  { code: "A5", label: "A5", condition: "A5_crop_only" },
];

export const ABLATION_CONDITION_BY_CODE: Record<AblationCode, string> = {
  A1: "A1_overlay_only",
  A2: "A2_overlay_confidence",
  A3: "A3_overlay_confidence_geometry",
  A4: "A4_overlay_crop_confidence",
  A5: "A5_crop_only",
};
