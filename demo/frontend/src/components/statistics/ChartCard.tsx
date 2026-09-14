import type { ReactNode } from "react";

export interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export function ChartCard({ title, subtitle, children, className = "" }: ChartCardProps) {
  return (
    <article className={`rounded border border-slate-200 bg-white ${className}`}>
      <header className="border-b border-slate-200 px-3 py-2">
        <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
        {subtitle ? <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p> : null}
      </header>
      <div className="flex min-h-[160px] flex-1 flex-col p-3">{children}</div>
    </article>
  );
}
