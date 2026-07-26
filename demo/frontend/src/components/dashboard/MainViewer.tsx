import type { SampleDetail } from "@shared/types";

import { ModelComparisonPanel } from "@/components/comparison";
import { OrthomosaicViewer } from "@/components/viewer";
import { MOCK_SAMPLE_COMPARISON } from "@/lib/mock/comparison";
import {
  MOCK_VIEWER_BOXES,
  MOCK_VIEWER_IMAGE,
} from "@/lib/mock/viewer";

interface MainViewerProps {
  sample: SampleDetail;
  activeModelKey?: string;
}

export function MainViewer({ sample, activeModelKey }: MainViewerProps) {
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
        imageWidth={MOCK_VIEWER_IMAGE.width}
        imageHeight={MOCK_VIEWER_IMAGE.height}
        boxes={MOCK_VIEWER_BOXES}
        initialSelectedId="yolo-primary"
        placeholderLabel={sample.image_name}
        interactive={false}
      />

      <ModelComparisonPanel
        className="max-h-[280px] shrink-0"
        data={MOCK_SAMPLE_COMPARISON}
        activeModelKey={activeModelKey}
      />
    </section>
  );
}
