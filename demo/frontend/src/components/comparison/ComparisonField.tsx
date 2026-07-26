import type { ReactNode } from "react";

export interface ComparisonFieldProps {
  label: string;
  children: ReactNode;
  mono?: boolean;
}

export function ComparisonField({ label, children, mono = false }: ComparisonFieldProps) {
  return (
    <div className="space-y-1">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </p>
      <div
        className={`text-xs leading-relaxed text-slate-300 ${mono ? "font-mono" : ""}`}
      >
        {children}
      </div>
    </div>
  );
}
