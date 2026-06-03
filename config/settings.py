"""
config/settings.py
------------------
Central configuration via Pydantic-Settings.
All values can be overridden with environment variables or a .env file.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API keys ──────────────────────────────────────────────────────────────
    gemini_api_key: SecretStr = Field(..., description="Google Gemini API key")
    serper_api_key: SecretStr = Field(..., description="Serper.dev API key")

    # ── Gemini model ──────────────────────────────────────────────────────────
    gemini_model: str = Field("gemini-1.5-flash", description="Gemini model ID")
    gemini_temperature: float = Field(0.2, ge=0.0, le=2.0)
    gemini_max_tokens: int = Field(4096, ge=256, le=32768)

    # ── Planner ───────────────────────────────────────────────────────────────
    max_sub_queries: int = Field(4, ge=1, le=8)

    # ── Search ────────────────────────────────────────────────────────────────
    serper_endpoint: str = "https://google.serper.dev/search"
    max_search_results: int = Field(5, ge=1, le=10)
    scrape_timeout_s: int = Field(15, ge=5, le=60)
    max_scraped_chars: int = Field(8_000, ge=1_000)

    # ── Orchestration ─────────────────────────────────────────────────────────
    pipeline_timeout_s: int = Field(120, ge=10, le=300)

    # ── Observability ─────────────────────────────────────────────────────────
    langsmith_api_key: str | None = Field(None)
    langchain_project: str = Field("perplexity-agent")
    log_level: str = Field("INFO")
    enable_otel: bool = Field(False)
    otel_endpoint: str = Field("http://localhost:4317")

    # ── FastAPI ───────────────────────────────────────────────────────────────
    app_host: str = Field("0.0.0.0")
    app_port: int = Field(8000, ge=1024, le=65535)
    app_workers: int = Field(1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


# Module-level singleton — import this everywhere
settings = Settings()
