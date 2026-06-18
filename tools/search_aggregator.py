"""
tools/search_aggregator.py
--------------------------
Multi-provider search aggregator:
  1. Serper (Google search via Serper.dev)
  2. Tavily  (purpose-built for LLM retrieval, pre-cleaned content)
  3. Firecrawl (deep scraping search, clean markdown)
  4. DuckDuckGo (free fallback)

Runs providers in parallel, deduplicates by domain,
and boosts authoritative domains.
"""
from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import structlog

from config.settings import settings
from models.schemas import SearchResult

log = structlog.get_logger()

# High-authority domains get score=1.0; everything else = 0.5
_HIGH_AUTHORITY = {
    "wikipedia.org", "arxiv.org", "github.com", "docs.python.org",
    "stackoverflow.com", "nature.com", "pubmed.ncbi.nlm.nih.gov",
    "docs.anthropic.com", "openai.com", "huggingface.co", "pytorch.org",
    "tensorflow.org", "developer.mozilla.org", "en.wikipedia.org",
    "sciencedirect.com", "researchgate.net", "scholar.google.com",
    "ncbi.nlm.nih.gov", "ieee.org", "acm.org", "readthedocs.io",
}


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host.replace("www.", "")
    except Exception:
        return url


def _authority_score(url: str) -> float:
    domain = extract_domain(url)
    # Check both exact domain and parent domain
    if domain in _HIGH_AUTHORITY:
        return 1.0
    parts = domain.split(".")
    if len(parts) >= 2:
        parent = ".".join(parts[-2:])
        if parent in _HIGH_AUTHORITY:
            return 1.0
    return 0.5


async def multi_provider_search(query: str, n: int | None = None) -> list[SearchResult]:
    """
    Run Serper + Tavily + Firecrawl in parallel.
    Falls back gracefully if any provider fails or is unconfigured.
    Deduplicates by domain and boosts authoritative results.
    """
    num = n or settings.max_search_results
    tasks = []
    labels = []
    
    # Enforce search timeout (max 6.0s) so slow providers fail fast and don't block
    search_timeout = min(getattr(settings, "scrape_timeout_s", 8.0), 6.0)

    # Always try Serper
    from tools.serper import serper_search
    tasks.append(asyncio.wait_for(serper_search(query, num=num), timeout=search_timeout))
    labels.append("serper")

    # Tavily if configured
    if settings.tavily_api_key:
        from tools.tavily_search import tavily_search
        tasks.append(asyncio.wait_for(tavily_search(query, num=num), timeout=search_timeout))
        labels.append("tavily")
    else:
        log.debug("tavily_skipped", reason="TAVILY_API_KEY not set")

    # Firecrawl if configured
    if settings.firecrawl_api_key:
        from tools.firecrawl_search import firecrawl_search
        tasks.append(asyncio.wait_for(firecrawl_search(query, num=num), timeout=search_timeout))
        labels.append("firecrawl")
    else:
        log.debug("firecrawl_skipped", reason="FIRECRAWL_API_KEY not set")

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge and deduplicate by domain
    seen_domains: set[str] = set()
    seen_urls: set[str] = set()
    merged: list[SearchResult] = []

    for label, results in zip(labels, raw_results):
        if isinstance(results, BaseException):
            log.warning("search_provider_failed", provider=label, error=str(results)[:120])
            continue
        for r in results:
            if not r.url:
                continue
            url_norm = r.url.rstrip("/").lower()
            domain = extract_domain(r.url)

            # Skip exact URL duplicates
            if url_norm in seen_urls:
                continue
            seen_urls.add(url_norm)

            # Allow max 2 results per domain for diversity
            domain_count = sum(1 for u in seen_urls if extract_domain(u) == domain)
            if domain_count > 2:
                continue

            seen_domains.add(domain)
            merged.append(r)

    # If all providers failed, try DuckDuckGo as last resort
    if not merged:
        log.warning("all_providers_failed_fallback_ddg", query=query[:60])
        try:
            from tools.duckduckgo import ddg_search
            merged = await ddg_search(query, num=num)
        except Exception as exc:
            log.error("ddg_fallback_failed", error=str(exc))

    # Sort: results with pre-scraped content first, then by authority
    merged.sort(
        key=lambda r: (
            1 if (r.scraped_text and len(r.scraped_text) > 200) else 0,
            _authority_score(r.url),
        ),
        reverse=True,
    )

    final = merged[:num]
    log.info("aggregator_done", query=query[:60], total=len(merged), returned=len(final))
    return final
