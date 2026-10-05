import type { Bbox, CellState, Condition, Degeneracy, SemanticLabel, VlmLabel } from "../types/api";

export const CONDITIONS: Condition[] = ["A1", "A2", "A3", "A4", "A5"];
export const VLM_LABELS: VlmLabel[] = ["Reliable", "Uncertain", "Unreliable"];
export const SEMANTIC_LABELS: SemanticLabel[] = ["palm", "ambiguous", "non_palm"];

export const VLM_SHORT: Record<VlmLabel, string> = { Reliable: "R", Uncertain: "U", Unreliable: "Ur" };

export const SEMANTIC_NAME: Record<SemanticLabel, string> = {
  palm: "Palm",
  ambiguous: "Ambiguous",
  non_palm: "Non-palm",
};

/**
 * Condition definitions as executed (outputs/verification_ablation_5747/<cond>/prompts and
 * prompt_index.csv image paths; crop construction in src/preprocessing/ablation_verification_images.py).
 */
export interface ConditionDef {
  code: Condition;
  dir: string;
  name: string;
  image: string;
  text: string;
  imageSameAs?: Condition;
}

export const CONDITION_DEFS: Record<Condition, ConditionDef> = {
  A1: {
    code: "A1",
    dir: "A1_overlay_only",
    name: "Overlay only",
    image: "Full 912×912 patch, background dimmed, green box on the detection.",
    text: "No detection metadata. Told not to use YOLO confidence or box geometry.",
  },
  A2: {
    code: "A2",
    dir: "A2_overlay_confidence",
    name: "Overlay + confidence",
    image: "Same image file as A1.",
    text: "Adds the YOLO confidence score as auxiliary text.",
    imageSameAs: "A1",
  },
  A3: {
    code: "A3",
    dir: "A3_overlay_confidence_geometry",
    name: "Overlay + confidence + geometry",
    image: "Same image file as A1.",
    text: "Adds YOLO confidence and box width, height, area and aspect ratio.",
    imageSameAs: "A1",
  },
  A4: {
    code: "A4",
    dir: "A4_overlay_crop_confidence",
    name: "Overlay + crop panel",
    image: "Two panels: the A1 overlay on the left, an enlarged crop of the box on the right.",
    text: "YOLO confidence as auxiliary text.",
  },
  A5: {
    code: "A5",
    dir: "A5_crop_only",
    name: "Crop only",
    image: "Only the box region (+15 px padding), resized to 512×512. No surrounding context.",
    text: "YOLO confidence as auxiliary text.",
  },
};

export const CELL_STATE_TEXT: Record<CellState, string> = {
  decision: "",
  parse_error: "parse error",
  missing: "missing",
  pending: "pending",
  not_run: "not run",
  not_audited: "not audited",
};

export const CELL_STATE_HELP: Record<CellState, string> = {
  decision: "",
  parse_error: "The model's output for this detection could not be parsed into a decision. Not repaired or imputed.",
  missing: "No valid decision in the analysis table for this cell.",
  pending: "Prediction files exist but the run is not finished or not yet audited; not shown.",
  not_run: "This model was not run on this condition (or the run is queued).",
  not_audited: "This model is not in the analysis audit inventory.",
};

export const DEGENERACY_TEXT: Record<Exclude<Degeneracy, null>, string> = {
  single_label: "single label",
  near_collapse_all: "near-collapse",
  near_collapse_some: "near-collapse (some conditions)",
};

export const DEGENERACY_HELP: Record<Exclude<Degeneracy, null>, string> = {
  single_label: "Every parseable decision in every usable condition is the same label.",
  near_collapse_all: "One label is ≥ 99% of decisions in every usable condition (descriptive cut-off, a10).",
  near_collapse_some: "One label is ≥ 99% of decisions in at least one usable condition (descriptive cut-off, a10).",
};

export const UNMATCHED_REASON_TEXT: Record<string, string> = {
  no_overlap: "no LabelMe palm box overlaps this detection",
  partial_overlap: "overlaps a LabelMe box, but IoU < 0.5",
  iou_ge_0_5_lost_greedy: "IoU ≥ 0.5 with a LabelMe box that a better-overlapping detection claimed",
  "iou_ge_0.5_lost_greedy": "IoU ≥ 0.5 with a LabelMe box that a better-overlapping detection claimed",
};

export const PAIR_GROUP_TEXT: Record<string, string> = {
  text_only: "text only",
  image_representation: "image representation",
  mixed_image_and_metadata: "image + metadata",
};

/** Crop window used by the project's blind review renderer (src/semantic_review/render.py). */
export function reviewCropWindow(bbox: Bbox, width: number, height: number): Bbox {
  const [x, y, w, h] = bbox;
  const side = Math.round(Math.min(Math.max(3 * Math.max(w, h), 160), width, height));
  const cx = x + w / 2;
  const cy = y + h / 2;
  const left = Math.round(Math.min(Math.max(cx - side / 2, 0), width - side));
  const top = Math.round(Math.min(Math.max(cy - side / 2, 0), height - side));
  return [left, top, side, side];
}

export function familyOrder(families: string[]): string[] {
  return Array.from(new Set(families));
}
