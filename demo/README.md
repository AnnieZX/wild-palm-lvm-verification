# Wild Palm Verification — Web Demo

Self-contained web demo for browsing verification **experiment outputs**. This folder is completely decoupled from the frozen verification pipeline (`src/`, `scripts/`, `configs/`, `outputs/`).

```
demo/
├── backend/     FastAPI REST API (future: read-only access to outputs server-side)
├── frontend/    Next.js + React + TypeScript + TailwindCSS
├── shared/      Cross-language schemas and interfaces
└── README.md
```

## Architecture rules

| Layer | Responsibility |
|-------|----------------|
| **Frontend** | UI only. Calls backend REST endpoints. Never reads `outputs/` directly. |
| **Backend** | Sole gateway to experiment artifacts. Exposes REST APIs only. |
| **Shared** | TypeScript interfaces + Pydantic models kept in sync. |

No business logic is implemented yet — the backend exposes a documented REST API with mock data in the final schema. See [`backend/API.md`](backend/API.md).

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness probe |
| GET | `/api/v1/models` | Model + experiment catalog |
| GET | `/api/v1/statistics` | Aggregate metrics for one ablation |
| GET | `/api/v1/samples` | Paginated sample list with filters |
| GET | `/api/v1/sample/{sample_id}` | Full sample detail |
| GET | `/api/v1/image/{sample_id}` | Overlay image (PNG) |

All experiment-scoped endpoints accept `model_key`, `experiment_id`, and `ablation` query parameters.

---

## Prerequisites

- **Node.js** 20+ and npm (for the frontend)
- **Python** 3.11+ (for the backend)

---

## Backend (FastAPI)

### Setup

```bash
cd demo/backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional — defaults work for local dev
```

### Run

From `demo/backend/` with the virtual environment activated:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify:

- Root: [http://localhost:8000/](http://localhost:8000/)
- Health: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- Models: [http://localhost:8000/api/v1/models](http://localhost:8000/api/v1/models)
- OpenAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Full endpoint reference: [`backend/API.md`](backend/API.md)

Environment variables (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEMO_API_HOST` | `0.0.0.0` | Bind address |
| `DEMO_API_PORT` | `8000` | Bind port |
| `DEMO_CORS_ORIGINS` | `http://localhost:3000` | Allowed frontend origins |
| `DEMO_OUTPUTS_ROOT` | _(unset)_ | Reserved for future read-only output access |

---

## Frontend (Next.js)

### Setup

```bash
cd demo/frontend
npm install
cp .env.local.example .env.local
```

### Run

From `demo/frontend/`:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The home page calls `GET /api/v1/health` on the backend.

Statistics dashboard (mock JSON, no API): [http://localhost:3000/statistics](http://localhost:3000/statistics)

Environment variables (see `.env.local.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Backend base URL for browser requests |

Other scripts:

```bash
npm run build      # production build
npm run start      # serve production build
npm run lint       # ESLint
npm run typecheck  # TypeScript check (includes ../shared/types.ts)
```

### UI theme

Design tokens live in `frontend/src/theme/` (forest green primary, slate neutrals, semantic orange/red/blue). See [`frontend/src/theme/README.md`](frontend/src/theme/README.md).

Tailwind classes: `forest-*`, `warning-*`, `error-*`, `selection-*`, utilities `wp-panel`, `wp-link`, `wp-label`.

### Dashboard layout

The home page is a scientific visualization dashboard (mock data only, no API wiring):

```
DashboardShell
├── Header
├── Sidebar
│   ├── ModelSelector
│   ├── PromptSelector
│   ├── ConfidenceFilter
│   ├── DecisionFilter
│   └── SampleSearch
├── MainViewer          (orthomosaic placeholder)
└── InformationPanel
    ├── YoloConfidenceCard
    ├── GroundTruthCard
    ├── PredictionCard
    ├── ReasoningCard
    └── ModelMetadataCard
```

Component roots: `frontend/src/components/dashboard/`, `sidebar/`, `info-panel/`.  
Mock fixtures: `frontend/src/lib/mock/dashboard.ts`.

---

## Running both services

Use two terminals:

**Terminal 1 — backend**

```bash
cd demo/backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — frontend**

```bash
cd demo/frontend
npm run dev
```

Start the backend first so the frontend health panel shows `status: ok`.

---

## Project layout

```
demo/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app + CORS
│   │   ├── config.py            Environment settings
│   │   └── api/
│   │       ├── router.py
│   │       └── routes/
│   │           └── health.py    GET /api/v1/health
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app/                 Next.js App Router pages
│   │   └── lib/
│   │       └── api.ts           Backend HTTP client (stub)
│   ├── package.json
│   └── .env.local.example
└── shared/
    ├── types.ts                 TypeScript interfaces
    ├── models.py                Pydantic models
    └── schemas/                 JSON Schema mirrors
```

---

## Next steps (not implemented)

- Backend endpoints to list experiments, samples, and metrics from `outputs/` (server-side only)
- Frontend pages for sample browsing and comparison views
- Authentication / deployment notes if needed

The verification pipeline remains unchanged.
