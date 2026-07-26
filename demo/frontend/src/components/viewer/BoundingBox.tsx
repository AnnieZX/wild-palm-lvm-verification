"use client";

import type { MouseEvent } from "react";

import { DEFAULT_BBOX_COLORS } from "@/components/viewer/constants";
import type {
  BoundingBoxColorScheme,
  BoundingBoxRole,
  ViewerBoundingBox,
} from "@/components/viewer/types";

export interface BoundingBoxProps {
  box: ViewerBoundingBox;
  imageWidth: number;
  imageHeight: number;
  isSelected?: boolean;
  isHovered?: boolean;
  colorScheme?: BoundingBoxColorScheme;
  /** When false, pointer events are disabled (scaffold default). */
  interactive?: boolean;
  onSelect?: (id: string) => void;
  onHoverStart?: (id: string, event: MouseEvent<SVGGElement>) => void;
  onHoverEnd?: (id: string) => void;
}

function resolveColors(
  role: BoundingBoxRole,
  isSelected: boolean,
  isHovered: boolean,
  scheme: BoundingBoxColorScheme,
) {
  if (isSelected) return scheme.selected;
  if (isHovered) return scheme.hover;
  return scheme[role === "ground-truth" ? "groundTruth" : role];
}

/**
 * SVG rectangle for one detection or GT box in image coordinate space.
 */
export function BoundingBox({
  box,
  imageWidth,
  imageHeight,
  isSelected = false,
  isHovered = false,
  colorScheme = DEFAULT_BBOX_COLORS,
  interactive = false,
  onSelect,
  onHoverStart,
  onHoverEnd,
}: BoundingBoxProps) {
  const colors = resolveColors(box.role, isSelected, isHovered, colorScheme);
  const label = box.label ?? box.role;

  return (
    <g
      data-bbox-id={box.id}
      data-bbox-role={box.role}
      className={interactive ? "cursor-pointer" : "pointer-events-none"}
      onClick={
        interactive && onSelect
          ? (event) => {
              event.stopPropagation();
              onSelect(box.id);
            }
          : undefined
      }
      onMouseEnter={
        interactive && onHoverStart
          ? (event) => onHoverStart(box.id, event)
          : undefined
      }
      onMouseLeave={
        interactive && onHoverEnd ? () => onHoverEnd(box.id) : undefined
      }
      aria-label={label}
    >
      <rect
        x={box.x}
        y={box.y}
        width={box.width}
        height={box.height}
        fill={colors.fill}
        stroke={colors.stroke}
        strokeWidth={colors.strokeWidth}
        vectorEffect="non-scaling-stroke"
        rx={2}
      />
      {(isSelected || isHovered) && (
        <text
          x={box.x + 4}
          y={Math.max(box.y - 6, 12)}
          fill={colors.stroke}
          fontSize={11}
          fontFamily="ui-monospace, monospace"
          className="select-none"
        >
          {label}
        </text>
      )}
      <title>
        {label} — {box.width}×{box.height} @ ({box.x}, {box.y}) in {imageWidth}×
        {imageHeight}
      </title>
    </g>
  );
}
