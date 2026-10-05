import type { ReactNode } from "react";

interface Props {
  id: string;
  kicker: string;
  title: ReactNode;
  lede?: ReactNode;
  children: ReactNode;
}

export function Section({ id, kicker, title, lede, children }: Props) {
  return (
    <section id={id} className="section">
      <div className="section-head">
        <div className="section-kicker">{kicker}</div>
        <div>
          <h2 className="section-title">{title}</h2>
          {lede && <div className="section-lede">{lede}</div>}
        </div>
      </div>
      {children}
    </section>
  );
}

export function Sources({ paths }: { paths: (string | undefined | null)[] }) {
  const list = paths.filter(Boolean) as string[];
  if (!list.length) return null;
  return <div className="source">{list.join("  ·  ")}</div>;
}

export function Loading({ what = "loading" }: { what?: string }) {
  return <div className="loading">{what}…</div>;
}

export function ErrorNote({ error }: { error: string }) {
  return <div className="error">Could not load: {error}</div>;
}
