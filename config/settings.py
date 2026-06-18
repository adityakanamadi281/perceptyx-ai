"""
config/settings.py - v2
Central configuration with Redis, PostgreSQL, Cloudflare AI fallback.
"""
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore",
        str_strip_whitespace=True,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    gemini_api_key: SecretStr = Field(..., description="Google Gemini API key")
    serper_api_key: SecretStr = Field(..., description="Serper.dev API key")

    # ── Cloudflare AI Fallback ────────────────────────────────────────────────
    cloudflare_account_id: str | None = Field(None)
    cloudflare_api_token: str | None = Field(None)
    cloudflare_model: str = Field("@cf/meta/llama-3.1-8b-instruct")
    cloudflare_gateway_id: str | None = Field(None)
    use_cloudflare_fallback: bool = Field(True)

    # ── Tavily ────────────────────────────────────────────────────────────────────
    tavily_api_key: SecretStr | None = Field(None, description="Tavily search API key")

    # ── Firecrawl ─────────────────────────────────────────────────────────────────
    firecrawl_api_key: SecretStr | None = Field(None, description="Firecrawl API key")

    # ── Optional API keys ─────────────────────────────────────────────────────
    newsapi_key: str | None = Field(None)
    gnews_api_key: str | None = Field(None)
    github_token: str | None = Field(None)

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_model: str = Field("gemini-2.5-flash")
    gemini_vision_model: str = Field("gemini-2.5-flash")
    gemini_temperature: float = Field(0.2, ge=0.0, le=2.0)
    gemini_max_tokens: int = Field(4096, ge=256, le=32768)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_url: str = Field("postgresql+asyncpg://perplexity:perplexity@localhost:5432/perplexity")
    postgres_pool_size: int = Field(10)
    postgres_max_overflow: int = Field(20)
    postgres_pool_timeout: int = Field(30)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379/0")
    redis_pool_size: int = Field(20)
    cache_query_ttl: int = Field(3600)
    cache_search_ttl: int = Field(1800)
    cache_answer_ttl: int = Field(7200)
    cache_session_ttl: int = Field(86400)

    # ── ARQ Workers ───────────────────────────────────────────────────────────
    arq_max_jobs: int = Field(50)
    arq_job_timeout: int = Field(120)
    worker_concurrency: int = Field(4)

    # ── Whisper ───────────────────────────────────────────────────────────────
    whisper_model_size: str = Field("base")
    whisper_device: str = Field("cpu")
    whisper_compute_type: str = Field("int8")

    # ── Planner ───────────────────────────────────────────────────────────────
    max_sub_queries: int = Field(4, ge=1, le=8)

    # ── Search ────────────────────────────────────────────────────────────────
    serper_endpoint: str = "https://google.serper.dev/search"
    max_search_results: int = Field(5, ge=1, le=10)
    scrape_timeout_s: int = Field(15, ge=5, le=60)
    max_scraped_chars: int = Field(8_000, ge=1_000)

    # ── News ──────────────────────────────────────────────────────────────────
    newsapi_endpoint: str = "https://newsapi.org/v2/everything"
    gnews_endpoint: str = "https://gnews.io/api/v4/search"
    max_news_results: int = Field(5, ge=1, le=20)

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_api_endpoint: str = "https://api.github.com"
    github_max_commits: int = Field(10)
    github_max_prs: int = Field(10)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field("http://localhost:6333")
    qdrant_api_key: SecretStr | None = Field(None, description="Qdrant API key")

    # ── Self-Learning & Parametric Knowledge ──────────────────────────────────
    enable_llm_knowledge: bool = Field(True)
    enable_self_learning: bool = Field(True)
    latency_budget_ms: int = Field(600)

    # ── RAG ───────────────────────────────────────────────────────────────────
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    rag_top_k: int = Field(5, ge=1, le=20)
    rag_retrieve_top_k: int = Field(50, ge=10, le=100)
    rag_score_threshold: float = Field(0.35)
    max_hops: int = Field(3, ge=1, le=5)

    # ── Reranking ─────────────────────────────────────────────────────────────
    enable_reranking: bool = Field(True)
    rerank_model: str = Field("BAAI/bge-reranker-base")
    rerank_top_k: int = Field(10)

    # ── Hybrid Search ─────────────────────────────────────────────────────────
    enable_hybrid_search: bool = Field(True)
    bm25_weight: float = Field(0.3)
    vector_weight: float = Field(0.7)

    # ── Router ────────────────────────────────────────────────────────────────
    router_recency_keywords: list[str] = Field(
        default_factory=lambda: [
            "today", "latest", "current", "recent", "news", "now", "2024", "2025", "2026"
        ]
    )

    # ── Query Complexity ─────────────────────────────────────────────────────
    complex_query_keywords: list[str] = Field(
        default_factory=lambda: [
            "compare","contrast","analyze","analyse","research","explain",
            "difference","versus","vs","pros and cons","tradeoffs",
            "comprehensive","in-depth","detailed","survey","overview",
        ]
    )

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_max_turns: int = Field(6, ge=1, le=20)
    memory_session_ttl_hours: int = Field(24)

    # ── Evaluation ────────────────────────────────────────────────────────────
    enable_eval: bool = Field(True)
    eval_faithfulness_threshold: float = Field(0.6)
    eval_only_research: bool = Field(True)

    # ── Orchestration ─────────────────────────────────────────────────────────
    pipeline_timeout_s: int = Field(120, ge=10, le=300)
    research_timeout_s: int = Field(300)

    # ── Observability ─────────────────────────────────────────────────────────
    langsmith_api_key: str | None = Field(None)
    langchain_project: str = Field("perplexity-agent-v2")
    log_level: str = Field("INFO")
    enable_otel: bool = Field(False)
    otel_endpoint: str = Field("http://localhost:4317")
    enable_prometheus: bool = Field(True)
    prometheus_port: int = Field(9100, ge=1024, le=65535)

    # ── FastAPI ───────────────────────────────────────────────────────────────
    app_host: str = Field("0.0.0.0")
    app_port: int = Field(8000, ge=1024, le=65535)
    app_workers: int = Field(1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Research mode ─────────────────────────────────────────────────────────
    research_max_queries: int = Field(8)
    research_crawl_urls: int = Field(5)
    research_gap_iterations: int = Field(2)


settings = Settings()
