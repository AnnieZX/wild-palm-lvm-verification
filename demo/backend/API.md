# Demo Backend API

REST API for serving **already-generated** verification experiment outputs.  
No model inference. No imports from `src/`. Filesystem reads are not implemented yet — all endpoints return mock data in the final response schema.

Base URL: `http://localhost:8000/api/v1`

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Endpoints

### `GET /health`

Liveness probe.

**Response:** `HealthResponse`

---

### `GET /models`

List VLM models and their experiment runs available in the demo catalog.

**Response:** `ModelsResponse`

```json
{
  "models": [
    {
      "model_key": "qwen2_5_vl",
      "display_name": "Qwen2.5-VL",
      "description": "...",
      "experiments": [
        {
          "experiment_id": "20250726_1200",
          "sample_count": 1000,
          "ablations": ["A1", "A2", "A3", "A4", "A5"],
          "primary_ablation": "A1",
          "created_at": "2025-07-26T12:00:00Z"
        }
      ]
    }
  ]
}
```

---

### `GET /statistics`

Aggregate metrics for one model × experiment × ablation (mirrors evaluation CSV / metrics JSON).

**Query parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_key` | string | `qwen2_5_vl` | Model registry key |
| `experiment_id` | string | `20250726_1200` | Experiment run ID |
| `ablation` | `A1`–`A5` | `A1` | Ablation code |

**Response:** `StatisticsResponse`

Includes precision, recall, F1, accuracy, IoU/confidence averages, decision distribution, and confusion counts.

---

### `GET /samples`

Paginated sample index with optional filters.

**Query parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_key` | string | `qwen2_5_vl` | Model registry key |
| `experiment_id` | string | `20250726_1200` | Experiment run ID |
| `ablation` | `A1`–`A5` | `A1` | Ablation code |
| `page` | int | `1` | Page number (1-based) |
| `page_size` | int | `20` | Page size (max 200) |
| `decision` | `Reliable` \| `Uncertain` \| `Unreliable` | — | Filter by VLM decision |
| `matched_gt` | bool | — | Filter by GT match |
| `gt_label` | `positive` \| `negative` \| `uncertain` | — | Filter by derived GT label |

**Response:** `SampleListResponse`

---

### `GET /sample/{sample_id}`

Full detail for one sample: bboxes, reasoning fields, and relative image API path.

**Path parameters:** `sample_id` (e.g. `sample_000042`)

**Query parameters:** `model_key`, `experiment_id`, `ablation` (same as above)

**Response:** `SampleDetailResponse`

---

### `GET /image/{sample_id}`

Ablation overlay image for a sample.

**Response:** `image/png` binary body

**Response headers (mock phase):**

- `X-Sample-Id`, `X-Model-Key`, `X-Experiment-Id`, `X-Ablation`
- `X-Mock-Image: true` — removed when real files are served

Future implementation will stream PNGs from ablation input directories server-side only.

---

## Schema location

| Layer | Path |
|-------|------|
| Pydantic | `demo/shared/models.py` |
| TypeScript | `demo/shared/types.ts` |
| Mock fixtures | `demo/backend/app/services/mock_data.py` |

When adding fields, update both `models.py` and `types.ts`.

---

## Error responses

All errors use `ApiError`:

```json
{ "detail": "Unknown sample_id: 'sample_999999'" }
```

HTTP status codes: `404` for unknown model/experiment/sample; `422` for invalid query parameters.
