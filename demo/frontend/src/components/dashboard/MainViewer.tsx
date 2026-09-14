import type { SampleDetail } from "@shared/types";

import { ModelComparisonPanel } from "@/components/comparison";
import type { SampleComparisonData } from "@/components/comparison/types";
import { OrthomosaicViewer } from "@/components/viewer";
import { sampleToViewerBoxes } from "@/lib/mappers";

interface MainViewerProps {
  sample: SampleDetail;
  activeModelKey?: string;
  imageSrc?: string | null;
  imageWidth: number;
  imageHeight: number;
  comparison: SampleComparisonData;
}

export function MainViewer({
  sample,
  activeModelKey,
  imageSrc = null,
  imageWidth,
  imageHeight,
  comparison,
}: MainViewerProps) {
  const boxes = sampleToViewerBoxes(sample);
  const metaLabel = `${sample.sample_id} · ${imageWidth}×${imageHeight}`;

  return (
    <section
      className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-[var(--wp-bg-viewer)]"
      aria-label="Aerial image inspector"
    >
      <div className="flex shrink-0 items-center justify-between border-b border-slate-800 px-3 py-1.5">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Aerial image
          </p>
          <p className="truncate font-mono text-xs text-slate-200">{sample.image_name}</p>
        </div>
        <p className="shrink-0 font-mono text-[11px] text-slate-500">{sample.sample_id}</p>
      </div>

      <OrthomosaicViewer
        className="min-h-0 flex-[2]"
        imageWidth={imageWidth}
        imageHeight={imageHeight}
        boxes={boxes}
        initialSelectedId={boxes[0]?.id ?? null}
        placeholderLabel={sample.image_name}
        imageSrc={imageSrc}
        interactive
        metaLabel={metaLabel}
      />

      <ModelComparisonPanel
        className="max-h-[220px] shrink-0"
        data={comparison}
        activeModelKey={activeModelKey}
      />
    </section>
  );
}
