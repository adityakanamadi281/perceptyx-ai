"""
workers/crawl_worker.py
───────────────────────
ARQ worker: fetch + extract clean text from URLs.
Uses httpx first, Playwright as JS-rendering fallback.
"""
from __future__ import annotations

import time
from typing import Any

import structlog

from config.settings import settings

log = structlog.get_logger()


async def crawl_urls(ctx: dict, urls: list[str]) -> list[dict]:
    """ARQ job: crawl_urls(urls) → list of {url, text, title, error}"""
    import asyncio
    results = await asyncio.gather(*[_crawl_one(url) for url in urls[:10]])
    return list(results)


async def _crawl_one(url: str) -> dict:
    t0 = time.perf_counter()
    try:
        import httpx
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient(timeout=settings.scrape_timeout_s, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 PerplexityBot/2.0"})
            resp.raise_for_status()
            html = resp.text
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title else ""
        text = " ".join(soup.get_text(" ", strip=True).split())
        text = text[:settings.max_scraped_chars]
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"url": url, "title": title, "text": text, "error": None, "latency_ms": latency_ms}
    except Exception as exc:
        log.warning("crawl_failed", url=url, error=str(exc))
        return {"url": url, "title": "", "text": "", "error": str(exc), "latency_ms": 0}
