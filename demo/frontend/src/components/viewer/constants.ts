import { colors } from "@/theme/tokens";
import type { BoundingBoxColorScheme, ViewerTransform } from "@/components/viewer/types";

/** Default stroke/fill palette aligned with demo theme tokens. */
export const DEFAULT_BBOX_COLORS: BoundingBoxColorScheme = {
  yolo: {
    stroke: colors.forest[400],
    fill: `${colors.forest[400]}1f`,
    strokeWidth: 2,
  },
  groundTruth: {
    stroke: colors.selection[400],
    fill: `${colors.selection[400]}1a`,
    strokeWidth: 2,
  },
  candidate: {
    stroke: colors.warning[400],
    fill: `${colors.warning[400]}1a`,
    strokeWidth: 1.5,
  },
  selected: {
    stroke: colors.selection[500],
    fill: `${colors.selection[500]}2e`,
    strokeWidth: 3,
  },
  hover: {
    stroke: colors.slate[200],
    fill: `${colors.slate[200]}14`,
    strokeWidth: 2.5,
  },
};

export const VIEWER_ZOOM_LIMITS = {
  min: 0.25,
  max: 8,
  step: 0.1,
} as const;

/** CSS transform string from ViewerTransform (applied to the scene layer). */
export function transformToStyle(transform: ViewerTransform): string {
  return `translate(${transform.translateX}px, ${transform.translateY}px) scale(${transform.scale})`;
}
