import { useState } from "react";
import type { Bbox } from "../../types/api";

interface Props {
  imageUrl: string;
  width: number;
  height: number;
  target: Bbox;
  /** Viewport in image pixels [x, y, w, h]; defaults to the full image. */
  window?: Bbox;
  others?: Bbox[];
  labelme?: Bbox[];
  targetColor?: string;
  showTarget?: boolean;
  label?: string;
}

/**
 * The raw source patch with boxes drawn as vector overlays in image-pixel coordinates,
 * so the YOLO box is never baked into a new image and always aligns with the source.
 */
export function PatchView({
  imageUrl,
  width,
  height,
  target,
  window,
  others = [],
  labelme = [],
  targetColor = "#ffd400",
  showTarget = true,
  label,
}: Props) {
  const [loaded, setLoaded] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const vb = window ?? [0, 0, width, height];
  const stroke = (px: number) => (px * vb[2]) / 600;
  const [x, y, w, h] = target;
  return (
    <svg viewBox={vb.join(" ")} preserveAspectRatio="xMidYMid meet" role="img" aria-label={label ?? "source patch"}>
      <rect x={vb[0]} y={vb[1]} width={vb[2]} height={vb[3]} fill="#111" />
      <image
        href={imageUrl}
        x={0}
        y={0}
        width={width}
        height={height}
        onLoad={() => setLoaded(imageUrl)}
        onError={() => setFailed(true)}
        style={{ opacity: loaded === imageUrl ? 1 : 0.0, transition: "opacity 160ms" }}
      />
      {loaded !== imageUrl && !failed && (
        <text x={vb[0] + vb[2] / 2} y={vb[1] + vb[3] / 2} fill="#888" fontSize={stroke(13)} textAnchor="middle" fontFamily="monospace">
          loading image…
        </text>
      )}
      {failed && (
        <text x={vb[0] + vb[2] / 2} y={vb[1] + vb[3] / 2} fill="#c88" fontSize={stroke(13)} textAnchor="middle" fontFamily="monospace">
          image unavailable
        </text>
      )}
      {others.map((b, i) => (
        <rect key={`o${i}`} x={b[0]} y={b[1]} width={b[2]} height={b[3]} fill="none" stroke="#fff" strokeOpacity={0.55} strokeWidth={stroke(1.2)} />
      ))}
      {labelme.map((b, i) => (
        <rect
          key={`l${i}`}
          x={b[0]}
          y={b[1]}
          width={Math.max(b[2], stroke(2))}
          height={Math.max(b[3], stroke(2))}
          fill="none"
          stroke="#5fd3ff"
          strokeWidth={stroke(1.8)}
          strokeDasharray={`${stroke(6)} ${stroke(4)}`}
        />
      ))}
      {showTarget && (
        <>
          <rect x={x} y={y} width={w} height={h} fill="none" stroke="#000" strokeOpacity={0.7} strokeWidth={stroke(4.5)} />
          <rect x={x} y={y} width={w} height={h} fill="none" stroke={targetColor} strokeWidth={stroke(2.4)} />
        </>
      )}
    </svg>
  );
}
