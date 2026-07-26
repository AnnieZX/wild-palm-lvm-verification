"""Backend configuration (environment variables only)."""

from __future__ import annotations

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
    demo_outputs_root: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.demo_cors_origins.split(",") if origin.strip()]


settings = Settings()
