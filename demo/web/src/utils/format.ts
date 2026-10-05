export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 0): string =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const fixed = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const ci = (lo: number | null | undefined, hi: number | null | undefined): string =>
  lo === null || lo === undefined || hi === null || hi === undefined ? "" : `[${pct(lo)}, ${pct(hi)}]`;

export const humanize = (s: string | null | undefined): string => (s ? s.replace(/_/g, " ") : "");
