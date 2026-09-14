interface HeaderProps {
  experimentId: string;
  sampleCount: number;
  modelName?: string;
  ablation?: string;
}

export function Header({
  experimentId,
  sampleCount,
  modelName,
  ablation,
}: HeaderProps) {
  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4">
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <h1 className="text-sm font-semibold tracking-tight text-slate-900">
            Wild Palm
          </h1>
          <span className="text-sm text-slate-400">/</span>
          <p className="text-sm text-slate-600">Detection Verification</p>
        </div>
      </div>

      <div className="hidden items-center gap-5 text-xs sm:flex">
        {modelName ? (
          <div className="text-right">
            <p className="text-slate-500">Model</p>
            <p className="font-medium text-slate-800">{modelName}</p>
          </div>
        ) : null}
        {ablation ? (
          <div className="text-right">
            <p className="text-slate-500">Ablation</p>
            <p className="font-mono font-medium text-slate-800">{ablation}</p>
          </div>
        ) : null}
        <div className="h-7 w-px bg-slate-200" aria-hidden />
        <div className="text-right">
          <p className="text-slate-500">Experiment</p>
          <p className="font-mono font-medium text-slate-800">{experimentId}</p>
        </div>
        <div className="text-right">
          <p className="text-slate-500">Samples</p>
          <p className="font-mono font-medium text-slate-800">
            {sampleCount.toLocaleString()}
          </p>
        </div>
      </div>
    </header>
  );
}
