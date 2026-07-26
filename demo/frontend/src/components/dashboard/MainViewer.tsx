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

  return (
    <section
      className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-slate-900"
      aria-label="Orthomosaic viewer"
    >
      <div className="flex shrink-0 items-center justify-between border-b border-slate-700/60 bg-slate-950/40 px-4 py-2">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Orthomosaic patch
          </p>
          <p className="font-mono text-sm text-slate-200">{sample.image_name}</p>
        </div>
        <p className="font-mono text-xs text-slate-500">{sample.sample_id}</p>
      </div>

      <OrthomosaicViewer
        className="min-h-0 flex-[2]"
        imageWidth={imageWidth}
        imageHeight={imageHeight}
        boxes={boxes}
        initialSelectedId={boxes[0]?.id ?? null}
        placeholderLabel={sample.image_name}
        imageSrc={imageSrc}
        interactive={false}
      />

      <ModelComparisonPanel
        className="max-h-[280px] shrink-0"
        data={comparison}
        activeModelKey={activeModelKey}
      />
    </section>
  );
}
