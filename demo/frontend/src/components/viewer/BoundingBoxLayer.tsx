"use client";

import type { MouseEvent } from "react";

import { BoundingBox } from "@/components/viewer/BoundingBox";
import { DEFAULT_BBOX_COLORS } from "@/components/viewer/constants";
import { useSelectionState } from "@/components/viewer/SelectionState";
import type {
  BoundingBoxColorScheme,
  HoverTooltipContent,
  ViewerBoundingBox,
  ViewerPointer,
  ViewerTransform,
} from "@/components/viewer/types";

export interface BoundingBoxLayerProps {
  boxes: ViewerBoundingBox[];
  imageWidth: number;
  imageHeight: number;
  transform?: ViewerTransform;
  colorScheme?: BoundingBoxColorScheme;
  /** Enable click / hover handlers (scaffold: default off). */
  interactive?: boolean;
  onTooltipChange?: (content: HoverTooltipContent | null, pointer: ViewerPointer | null) => void;
}

function buildTooltipContent(box: ViewerBoundingBox): HoverTooltipContent {
  const lines = [
    `role: ${box.role}`,
    `x=${box.x} y=${box.y}`,
    `w=${box.width} h=${box.height}`,
  ];
  if (box.meta) {
    for (const [key, value] of Object.entries(box.meta)) {
      lines.push(`${key}: ${value}`);
    }
  }
  return { title: box.label ?? box.id, lines };
}

/**
 * SVG overlay layer rendering all bounding boxes in image space.
 */
export function BoundingBoxLayer({
  boxes,
  imageWidth,
  imageHeight,
  transform: _transform,
  colorScheme = DEFAULT_BBOX_COLORS,
  interactive = false,
  onTooltipChange,
}: BoundingBoxLayerProps) {
  const { selectedId, hoveredId, select, hover } = useSelectionState();

  const handleHoverStart = (id: string, event: MouseEvent<SVGGElement>) => {
    hover(id);
    const box = boxes.find((item) => item.id === id);
    if (box && onTooltipChange) {
      const svg = event.currentTarget.ownerSVGElement;
      const rect = svg?.getBoundingClientRect();
      if (rect) {
        onTooltipChange(buildTooltipContent(box), {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
        });
      }
    }
  };

  const handleHoverEnd = (id: string) => {
    if (hoveredId === id) {
      hover(null);
      onTooltipChange?.(null, null);
    }
  };

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden={!interactive}
    >
      <g className={interactive ? "pointer-events-auto" : ""}>
        {boxes.map((box) => (
          <BoundingBox
            key={box.id}
            box={box}
            imageWidth={imageWidth}
            imageHeight={imageHeight}
            isSelected={selectedId === box.id}
            isHovered={hoveredId === box.id}
            colorScheme={colorScheme}
            interactive={interactive}
            onSelect={select}
            onHoverStart={handleHoverStart}
            onHoverEnd={handleHoverEnd}
          />
        ))}
      </g>
    </svg>
  );
}
