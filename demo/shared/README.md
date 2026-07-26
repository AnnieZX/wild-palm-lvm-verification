# Shared schemas

Cross-language contracts between the demo frontend and backend.

| File | Purpose |
|------|---------|
| `types.ts` | TypeScript interfaces consumed by the Next.js app |
| `models.py` | Pydantic models consumed by the FastAPI backend |
| `schemas/*.json` | Optional JSON Schema mirrors for documentation |

Keep `types.ts` and `models.py` in sync when adding new API payloads.
See `demo/backend/API.md` for the full endpoint catalog.

The verification pipeline under `src/` is not imported here.
