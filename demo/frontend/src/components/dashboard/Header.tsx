interface HeaderProps {
  experimentId: string;
  sampleCount: number;
}

export function Header({ experimentId, sampleCount }: HeaderProps) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200/80 bg-white px-5 shadow-sm">
      <div className="flex items-center gap-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-forest-700 text-sm font-bold text-white">
          WP
        </div>
        <div>
          <h1 className="text-base font-semibold tracking-tight text-slate-900">
            Wild Palm Verification Demo
          </h1>
          <p className="text-xs text-slate-500">
            Scientific visualization dashboard · layout prototype
          </p>
        </div>
      </div>

      <div className="hidden items-center gap-6 text-xs sm:flex">
        <div className="text-right">
          <p className="text-slate-500">Experiment</p>
          <p className="font-mono font-medium text-slate-800">{experimentId}</p>
        </div>
        <div className="h-8 w-px bg-slate-200" aria-hidden />
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
