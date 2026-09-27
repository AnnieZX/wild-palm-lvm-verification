"""Backend configuration (environment variables only)."""

from __future__ import annotations

from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    demo_api_host: str = "0.0.0.0"
    demo_api_port: int = 8000
    # Comma-separated browser origins for CORS. Override in production via DEMO_CORS_ORIGINS.
    demo_cors_origins: str = (
        "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:3003"
    )
    demo_outputs_root: Optional[str] = None
    # Evaluation tree, absolute or relative to the outputs root. Default: evaluation_protocol_v2
    # (Evaluation Protocol v2). Set DEMO_EVALUATION_ROOT=evaluation to browse frozen Protocol v1.
    demo_evaluation_root: Optional[str] = None

    @property
    def outputs_root(self):
        from pathlib import Path

        if self.demo_outputs_root:
            return Path(self.demo_outputs_root).resolve()
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "outputs"

    @property
    def evaluation_root(self):
        from pathlib import Path

        return Path(self.demo_evaluation_root) if self.demo_evaluation_root else None

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.demo_cors_origins.split(",") if origin.strip()]


settings = Settings()
