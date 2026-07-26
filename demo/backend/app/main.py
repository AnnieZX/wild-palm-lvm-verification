"""
FastAPI entrypoint for the wild palm verification web demo.

This service is decoupled from the frozen verification pipeline.
Future endpoints may read experiment artifacts server-side only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Allow `from shared.models import ...` when running from demo/backend.
DEMO_ROOT = Path(__file__).resolve().parents[2]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from app.api.router import api_router  # noqa: E402
from app.config import settings  # noqa: E402

app = FastAPI(
    title="Wild Palm Verification Demo API",
    description="REST API for browsing verification experiment outputs.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Wild Palm Verification Demo API"}
