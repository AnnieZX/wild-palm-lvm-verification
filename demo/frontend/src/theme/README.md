# Wild Palm Demo — UI Theme

Professional, minimal theme for academic reviewers and remote sensing practitioners.

## Token sources

| File | Purpose |
|------|---------|
| `src/theme/tokens.ts` | Color scales, typography, layout (programmatic) |
| `src/theme/decision-styles.ts` | Verification decision badge/chart classes |
| `tailwind.config.ts` | Tailwind `forest`, `warning`, `error`, `selection` colors |
| `src/app/globals.css` | CSS custom properties + component utilities |

## Color semantics

| Token | Use |
|-------|-----|
| `forest-*` | Primary brand, correct / Reliable decisions |
| `slate-*` | Neutrals, text, borders, viewer chrome |
| `surface` | White panel backgrounds |
| `warning-*` | Uncertain decisions, cautions |
| `error-*` | Unreliable / incorrect detections |
| `selection-*` | Active selection, highlighted model card |

## Tailwind examples

```tsx
<div className="wp-panel">...</div>
<span className="bg-forest-700 text-white">Primary</span>
<span className="bg-warning-100 text-warning-800">Uncertain</span>
<span className="bg-error-100 text-error-800">Unreliable</span>
<span className="ring-selection-500/40 border-selection-500/60">Selected</span>
<a className="wp-link" href="/statistics">Statistics</a>
```

## Decision badges

```tsx
import { decisionBadgeLight } from "@/theme/decision-styles";

<span className={`ring-1 ring-inset ${decisionBadgeLight.Reliable}`}>Reliable</span>
```

No animations are defined in the theme — keep interactions static and scientific.
