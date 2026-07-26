import type { ReactNode } from "react";

interface InfoCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export function InfoCard({ title, subtitle, children, className = "" }: InfoCardProps) {
  return (
    <article
      className={`wp-panel p-4 ${className}`}
    >
      <header className="mb-3 border-b border-slate-100 pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {title}
        </h3>
        {subtitle ? (
          <p className="mt-0.5 text-[11px] text-slate-400">{subtitle}</p>
        ) : null}
      </header>
      <div className="text-sm text-slate-700">{children}</div>
    </article>
  );
}
