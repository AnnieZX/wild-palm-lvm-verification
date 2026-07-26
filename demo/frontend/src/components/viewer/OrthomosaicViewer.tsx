"use client";

import { useRef, useState, type ReactNode } from "react";

import { BoundingBoxLayer } from "@/components/viewer/BoundingBoxLayer";
import {
  DEFAULT_BBOX_COLORS,
  transformToStyle,
  VIEWER_ZOOM_LIMITS,
} from "@/components/viewer/constants";
import { HoverTooltip } from "@/components/viewer/HoverTooltip";
import { SelectionStateProvider } from "@/components/viewer/SelectionState";
import type {
  BoundingBoxColorScheme,
  HoverTooltipContent,
  ViewerBoundingBox,
  ViewerPointer,
  ViewerTransform,
} from "@/components/viewer/types";
import { DEFAULT_VIEWER_TRANSFORM } from "@/components/viewer/types";

export interface OrthomosaicViewerProps {
  /** Image coordinate space width (pixels). */
  imageWidth: number;
  /** Image coordinate space height (pixels). */
  imageHeight: number;
  boxes?: ViewerBoundingBox[];
  colorScheme?: BoundingBoxColorScheme;
  initialSelectedId?: string | null;
  placeholderLabel?: string;
  /** Future: URL from backend — not fetched in this scaffold. */
  imageSrc?: string | null;
  /** Enable bbox click / hover (scaffold default: false). */
  interactive?: boolean;
  toolbar?: ReactNode;
  className?: string;
}

/**
 * Orthomosaic patch viewer with layered bbox overlay.
 * Image loading, zoom, and pan are stubbed for future implementation.
 */
export function OrthomosaicViewer({
  imageWidth,
  imageHeight,
  boxes = [],
  colorScheme = DEFAULT_BBOX_COLORS,
  initialSelectedId = null,
  placeholderLabel = "Orthomosaic image",
  imageSrc = null,
  interactive = false,
  toolbar,
  className = "",
}: OrthomosaicViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState<ViewerTransform>(DEFAULT_VIEWER_TRANSFORM);
  const [tooltipContent, setTooltipContent] = useState<HoverTooltipContent | null>(null);
  const [tooltipPointer, setTooltipPointer] = useState<ViewerPointer | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  const handleTooltipChange = (
    content: HoverTooltipContent | null,
    pointer: ViewerPointer | null,
  ) => {
    setTooltipContent(content);
    setTooltipPointer(pointer);
  };

  // Reserved for future zoom / pan handlers on viewportRef.
  void setTransform;
  void VIEWER_ZOOM_LIMITS;

  return (
    <SelectionStateProvider initialSelectedId={initialSelectedId}>
      <div className={`flex h-full min-h-0 flex-col ${className}`}>
        {toolbar ? (
          <div className="shrink-0 border-b border-slate-700/60 bg-slate-950/40 px-3 py-2">
            {toolbar}
          </div>
        ) : null}

        <div
          ref={(node) => {
            viewportRef.current = node;
            if (node) setContainerWidth(node.clientWidth);
          }}
          className="relative min-h-0 flex-1 overflow-hidden bg-slate-900"
          data-viewer-viewport
          aria-label="Orthomosaic viewport"
        >
          <div
            className="absolute inset-0 flex items-center justify-center p-4"
            style={{ transform: transformToStyle(transform), transformOrigin: "center center" }}
            data-viewer-scene
          >
            <div
              className="relative aspect-square w-full max-h-full max-w-full overflow-hidden rounded-lg border border-slate-600/80 bg-slate-800 shadow-inner"
              style={{ aspectRatio: `${imageWidth} / ${imageHeight}` }}
            >
              <div
                className="absolute inset-0 dashboard-grid-bg bg-slate-800/90"
                data-viewer-image-layer
              >
                {imageSrc ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={imageSrc}
                    alt={placeholderLabel}
                    className="hidden h-full w-full object-contain"
                    data-viewer-image
                  />
                ) : (
                  <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                    <span className="text-xs font-medium uppercase tracking-widest text-slate-500">
                      {placeholderLabel}
                    </span>
                    <span className="font-mono text-[10px] text-slate-600">
                      {imageWidth} × {imageHeight}px
                    </span>
                  </div>
                )}
              </div>

              <BoundingBoxLayer
                boxes={boxes}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                transform={transform}
                colorScheme={colorScheme}
                interactive={interactive}
                onTooltipChange={handleTooltipChange}
              />
            </div>
          </div>

          <HoverTooltip
            visible={interactive && tooltipContent !== null}
            pointer={tooltipPointer}
            content={tooltipContent}
            containerWidth={containerWidth}
          />
        </div>

        <footer className="shrink-0 border-t border-slate-700/60 bg-slate-950/40 px-3 py-1.5">
          <p className="text-center font-mono text-[10px] text-slate-500">
            zoom · pan · selection — scaffold ({Math.round(transform.scale * 100)}%)
          </p>
        </footer>
      </div>
    </SelectionStateProvider>
  );
}
