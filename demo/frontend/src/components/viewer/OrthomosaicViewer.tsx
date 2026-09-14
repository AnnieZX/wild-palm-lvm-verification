"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { BoundingBoxLayer } from "@/components/viewer/BoundingBoxLayer";
import {
  DEFAULT_BBOX_COLORS,
  transformToStyle,
} from "@/components/viewer/constants";
import { HoverTooltip } from "@/components/viewer/HoverTooltip";
import { SelectionStateProvider } from "@/components/viewer/SelectionState";
import type {
  BoundingBoxColorScheme,
  HoverTooltipContent,
  ViewerBoundingBox,
  ViewerPointer,
} from "@/components/viewer/types";
import { DEFAULT_VIEWER_TRANSFORM } from "@/components/viewer/types";

export interface OrthomosaicViewerProps {
  imageWidth: number;
  imageHeight: number;
  boxes?: ViewerBoundingBox[];
  colorScheme?: BoundingBoxColorScheme;
  initialSelectedId?: string | null;
  placeholderLabel?: string;
  imageSrc?: string | null;
  interactive?: boolean;
  toolbar?: ReactNode;
  className?: string;
  metaLabel?: string;
}

/**
 * Inspection canvas for aerial patch imagery with bbox overlay.
 * Fits image to viewport; no fake pan/zoom controls.
 */
export function OrthomosaicViewer({
  imageWidth,
  imageHeight,
  boxes = [],
  colorScheme = DEFAULT_BBOX_COLORS,
  initialSelectedId = null,
  placeholderLabel = "Aerial patch",
  imageSrc = null,
  interactive = true,
  toolbar,
  className = "",
  metaLabel,
}: OrthomosaicViewerProps) {
  const [tooltipContent, setTooltipContent] = useState<HoverTooltipContent | null>(null);
  const [tooltipPointer, setTooltipPointer] = useState<ViewerPointer | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [imageFailed, setImageFailed] = useState(false);
  const transform = DEFAULT_VIEWER_TRANSFORM;
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  useEffect(() => {
    setImageFailed(false);
  }, [imageSrc]);

  const handleTooltipChange = (
    content: HoverTooltipContent | null,
    pointer: ViewerPointer | null,
  ) => {
    setTooltipContent(content);
    setTooltipPointer(pointer);
  };

  return (
    <SelectionStateProvider initialSelectedId={initialSelectedId}>
      <div className={`flex h-full min-h-0 flex-col ${className}`}>
        {toolbar ? (
          <div className="shrink-0 border-b border-slate-700/60 bg-slate-950/50 px-3 py-1.5">
            {toolbar}
          </div>
        ) : null}

        <div
          ref={(node) => {
            if (resizeObserverRef.current) {
              resizeObserverRef.current.disconnect();
              resizeObserverRef.current = null;
            }
            if (!node) {
              return;
            }
            setContainerWidth(node.clientWidth);
            const observer = new ResizeObserver((entries) => {
              const entry = entries[0];
              if (entry) {
                setContainerWidth(entry.contentRect.width);
              }
            });
            observer.observe(node);
            resizeObserverRef.current = observer;
          }}
          className="relative min-h-0 flex-1 overflow-hidden bg-[var(--wp-bg-viewer)]"
          data-viewer-viewport
          aria-label="Aerial inspection canvas"
        >
          <div
            className="absolute inset-0 flex items-center justify-center p-3"
            style={{ transform: transformToStyle(transform), transformOrigin: "center center" }}
            data-viewer-scene
          >
            <div
              className="relative h-full max-h-full w-auto max-w-full overflow-hidden border border-slate-700/80 bg-black"
              style={{ aspectRatio: `${imageWidth} / ${imageHeight}` }}
            >
              <div className="absolute inset-0 bg-black" data-viewer-image-layer>
                {imageSrc && !imageFailed ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={imageSrc}
                    alt={placeholderLabel}
                    className="h-full w-full object-contain"
                    data-viewer-image
                    onError={() => setImageFailed(true)}
                  />
                ) : (
                  <div className="flex h-full flex-col items-center justify-center gap-1 text-center">
                    <span className="font-mono text-xs text-slate-500">
                      {imageFailed ? "Image unavailable" : "No image"}
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

          {metaLabel ? (
            <div className="pointer-events-none absolute bottom-2 left-2 rounded border border-slate-700/70 bg-slate-950/80 px-2 py-1 font-mono text-[10px] text-slate-300">
              {metaLabel}
            </div>
          ) : null}
        </div>
      </div>
    </SelectionStateProvider>
  );
}
