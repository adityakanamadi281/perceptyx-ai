"""
tools/news.py
-------------
News retrieval via NewsAPI and GNews with automatic fallback.
NewsAPI is tried first; GNews is used if NewsAPI key is absent or fails.
"""

from __future__ import annotations

import time

import httpx
import structlog

from config.settings import settings
from models.schemas import NewsArticle, NewsOutput

log = structlog.get_logger()


async def _fetch_newsapi(query: str, n: int) -> list[NewsArticle]:
    if not settings.newsapi_key:
        raise ValueError("NewsAPI key not configured")

    params = {
        "q": query,
        "pageSize": n,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": settings.newsapi_key,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(settings.newsapi_endpoint, params=params)
        resp.raise_for_status()
        data = resp.json()

    articles = []
    for a in data.get("articles", [])[:n]:
        articles.append(NewsArticle(
            title=a.get("title", ""),
            url=a.get("url", ""),
            source_name=a.get("source", {}).get("name", ""),
            published_at=a.get("publishedAt"),
            description=a.get("description"),
            content=a.get("content"),
            provider="newsapi",
        ))
    return articles


async def _fetch_gnews(query: str, n: int) -> list[NewsArticle]:
    if not settings.gnews_api_key:
        raise ValueError("GNews API key not configured")

    params = {
        "q": query,
        "max": n,
        "lang": "en",
        "token": settings.gnews_api_key,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(settings.gnews_endpoint, params=params)
        resp.raise_for_status()
        data = resp.json()

    articles = []
    for a in data.get("articles", [])[:n]:
        articles.append(NewsArticle(
            title=a.get("title", ""),
            url=a.get("url", ""),
            source_name=a.get("source", {}).get("name", ""),
            published_at=a.get("publishedAt"),
            description=a.get("description"),
            content=a.get("content"),
            provider="gnews",
        ))
    return articles


async def fetch_news(query: str, sub_query: str | None = None) -> NewsOutput:
    """
    Fetch news articles. Tries NewsAPI first, falls back to GNews.
    Returns a NewsOutput with the merged articles.
    """
    t0 = time.perf_counter()
    n = settings.max_news_results
    q = sub_query or query
    articles: list[NewsArticle] = []

    try:
        articles = await _fetch_newsapi(q, n)
        log.info("newsapi_ok", count=len(articles))
    except Exception as e1:
        log.warning("newsapi_failed", error=str(e1))
        try:
            articles = await _fetch_gnews(q, n)
            log.info("gnews_ok", count=len(articles))
        except Exception as e2:
            log.error("both_news_failed", error=str(e2))
            try:
                from tools.serper import serper_search
                log.info("news_fallback_web_start", query=q)
                web_results = await serper_search(q, n)
                for r in web_results:
                    articles.append(NewsArticle(
                        title=r.title,
                        url=r.url,
                        source_name="Web Search",
                        description=r.snippet,
                        provider="newsapi",
                    ))
                log.info("news_fallback_web_ok", count=len(articles))
            except Exception as e3:
                log.error("news_fallback_web_failed", error=str(e3))

    return NewsOutput(
        sub_query=q,
        articles=articles,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
