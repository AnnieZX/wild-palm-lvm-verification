/**
 * Viewer coordinate space and interaction types.
 * Independent from backend API shapes — map SampleDetail bboxes at the integration layer.
 */

/** Semantic role for color coding and layer ordering. */
export type BoundingBoxRole = "yolo" | "ground-truth" | "candidate";

/** Single box in image pixel coordinates (origin top-left). */
export interface ViewerBoundingBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  role: BoundingBoxRole;
  label?: string;
  /** Optional payload for tooltips without backend coupling. */
  meta?: Record<string, string | number | boolean>;
}

/** Viewport transform for future zoom / pan. */
export interface ViewerTransform {
  scale: number;
  translateX: number;
  translateY: number;
}

export const DEFAULT_VIEWER_TRANSFORM: ViewerTransform = {
  scale: 1,
  translateX: 0,
  translateY: 0,
};

/** Selection + hover ids tracked by the viewer. */
export interface ViewerSelectionState {
  selectedId: string | null;
  hoveredId: string | null;
}

export interface SelectionStateActions {
  select: (id: string | null) => void;
  hover: (id: string | null) => void;
  clear: () => void;
}

/** Color tokens per visual state (CSS color values). */
export interface BoundingBoxColorScheme {
  yolo: BoxStrokeFill;
  groundTruth: BoxStrokeFill;
  candidate: BoxStrokeFill;
  selected: BoxStrokeFill;
  hover: BoxStrokeFill;
}

export interface BoxStrokeFill {
  stroke: string;
  fill: string;
  strokeWidth: number;
}

/** Pointer position in viewer-local coordinates for tooltip placement. */
export interface ViewerPointer {
  x: number;
  y: number;
}

export interface HoverTooltipContent {
  title: string;
  lines?: string[];
}
