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
        panel: "none",
      },
      borderRadius: {
        panel: "0.375rem",
      },
    },
  },
  plugins: [],
};

export default config;
