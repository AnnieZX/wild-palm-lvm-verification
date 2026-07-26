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
    demo_cors_origins: str = "http://localhost:3000"
    demo_outputs_root: Optional[str] = None

    @property
    def outputs_root(self):
        from pathlib import Path

        if self.demo_outputs_root:
            return Path(self.demo_outputs_root).resolve()
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "outputs"

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.demo_cors_origins.split(",") if origin.strip()]


settings = Settings()
