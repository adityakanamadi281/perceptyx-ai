"""
workers/settings.py
───────────────────
ARQ worker pool configuration.
"""
from __future__ import annotations

from arq.connections import RedisSettings
from config.settings import settings as app_settings


class WorkerSettings:
    """ARQ WorkerSettings — passed to `arq worker workers.settings.WorkerSettings`"""
    functions = [
        "workers.search_worker.search_web",
        "workers.crawl_worker.crawl_urls",
        "workers.embedding_worker.embed_documents",
        "workers.research_worker.deep_research",
        "workers.evaluation_worker.evaluate_async",
    ]
    redis_settings = RedisSettings.from_dsn(app_settings.redis_url)
    max_jobs = app_settings.arq_max_jobs
    job_timeout = app_settings.arq_job_timeout
    keep_result = 3600  # keep job results for 1 hour
