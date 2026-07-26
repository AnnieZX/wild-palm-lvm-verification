/**
 * Wild Palm Verification Demo — design tokens.
 *
 * Academic / scientific UI: forest green primary, slate neutrals,
 * semantic orange (warning), red (error), blue (selection).
 */

export const colors = {
  forest: {
    50: "#f4f7f5",
    100: "#e4ece6",
    200: "#c9d9cd",
    300: "#a3bfaa",
    400: "#729a7e",
    500: "#4f7a5c",
    600: "#3d6249",
    700: "#32503c",
    800: "#2a4032",
    900: "#233529",
    950: "#121c16",
  },
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617",
  },
  white: "#ffffff",
  warning: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
  },
  error: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
  },
  selection: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
  },
} as const;

export const typography = {
  fontSans: [
    "Inter",
    "ui-sans-serif",
    "system-ui",
    "-apple-system",
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
  ],
  fontMono: [
    "ui-monospace",
    "SFMono-Regular",
    "Menlo",
    "Monaco",
    "Consolas",
    "Liberation Mono",
    "Courier New",
    "monospace",
  ],
} as const;

export const radii = {
  sm: "0.375rem",
  md: "0.5rem",
  lg: "0.75rem",
  xl: "1rem",
} as const;

export const shadows = {
  sm: "0 1px 2px 0 rgb(15 23 42 / 0.05)",
  md: "0 4px 6px -1px rgb(15 23 42 / 0.07), 0 2px 4px -2px rgb(15 23 42 / 0.05)",
  panel: "0 1px 3px 0 rgb(15 23 42 / 0.08)",
} as const;

export const layout = {
  sidebarWidth: "280px",
  infoPanelWidth: "320px",
  headerHeight: "3.5rem",
} as const;

/** Semantic aliases for documentation and programmatic use. */
export const semantic = {
  primary: colors.forest,
  neutral: colors.slate,
  surface: colors.white,
  warning: colors.warning,
  error: colors.error,
  selection: colors.selection,
  correct: colors.forest,
  incorrect: colors.error,
} as const;
