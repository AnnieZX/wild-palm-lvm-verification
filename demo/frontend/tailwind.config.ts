import type { Config } from "tailwindcss";

import { colors as themeColors, typography } from "./src/theme/tokens";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: themeColors.forest,
        warning: themeColors.warning,
        error: themeColors.error,
        selection: themeColors.selection,
        surface: themeColors.white,
        /** @deprecated Use `forest` — alias kept for gradual migration */
        palm: themeColors.forest,
      },
      fontFamily: {
        sans: [...typography.fontSans],
        mono: [...typography.fontMono],
      },
      boxShadow: {
        panel: "0 1px 3px 0 rgb(15 23 42 / 0.08)",
      },
      borderRadius: {
        panel: "0.75rem",
      },
    },
  },
  plugins: [],
};

export default config;
