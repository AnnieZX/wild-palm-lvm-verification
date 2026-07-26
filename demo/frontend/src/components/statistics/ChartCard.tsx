import type { ReactNode } from "react";

export interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export function ChartCard({ title, subtitle, children, className = "" }: ChartCardProps) {
  return (
    <article
      className={`wp-panel overflow-hidden ${className}`}
    >
      <header className="border-b border-slate-100 px-4 py-3">
        <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
        {subtitle ? <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p> : null}
      </header>
      <div className="flex min-h-[200px] flex-1 flex-col p-4">{children}</div>
    </article>
  );
}
