"use client";

import type { HoverTooltipContent, ViewerPointer } from "@/components/viewer/types";

export interface HoverTooltipProps {
  visible: boolean;
  pointer: ViewerPointer | null;
  content: HoverTooltipContent | null;
  /** Viewer container width — used to flip tooltip away from edges (future). */
  containerWidth?: number;
}

/**
 * Floating tooltip for bbox hover metadata.
 * Positioned absolutely within the viewer viewport.
 */
export function HoverTooltip({
  visible,
  pointer,
  content,
  containerWidth,
}: HoverTooltipProps) {
  if (!visible || !pointer || !content) {
    return null;
  }

  const flipLeft = containerWidth !== undefined && pointer.x > containerWidth * 0.65;

  return (
    <div
      className="pointer-events-none absolute z-30 max-w-[220px] rounded-md border border-slate-600/80 bg-slate-950/95 px-3 py-2 shadow-lg backdrop-blur-sm"
      style={{
        left: pointer.x + (flipLeft ? -12 : 12),
        top: pointer.y + 12,
        transform: flipLeft ? "translateX(-100%)" : undefined,
      }}
      role="tooltip"
    >
      <p className="text-xs font-semibold text-slate-100">{content.title}</p>
      {content.lines?.length ? (
        <ul className="mt-1.5 space-y-0.5">
          {content.lines.map((line) => (
            <li key={line} className="font-mono text-[10px] text-slate-400">
              {line}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
