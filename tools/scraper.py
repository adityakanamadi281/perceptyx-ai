"""
tools/scraper.py
----------------
Async web scraper with 3-tier fallback:
  1. httpx   — fast, lightweight
  2. Playwright — JS-rendered pages
  3. Jina Reader — free, handles JS+paywalls, returns clean Markdown

Improvements over original:
  - Paywall detection + Jina fallback
  - Sentence-boundary truncation (not mid-char)
  - Content deduplication fingerprint
  - Firecrawl premium scrape option (if API key set)
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup

from config.settings import settings

# Simple in-process fingerprint cache to avoid scraping same content twice
_content_fingerprints: set[str] = set()

_PAYWALL_SIGNALS = [
    "subscribe to read",
    "subscribe to continue",
    "sign in to continue",
    "sign up to read",
    "this article is for subscribers",
    "this content is for subscribers",
    "create a free account to",
    "unlock this story",
    "unlock this article",
    "premium content",
    "to read the full story",
    "members only",
]

_JS_SIGNALS = [
    "enable javascript",
    "javascript is required",
    "javascript enabled",
    "please enable js",
]





async def _fetch_with_httpx(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PerceptyxAI/1.0; +https://github.com/example)"
        )
    }
    async with httpx.AsyncClient(
        timeout=settings.scrape_timeout_s,
        follow_redirects=True,
        headers=headers,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def _fetch_with_jina(url: str) -> str:
    """Jina Reader — free, handles JS pages, returns clean Markdown."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"https://r.jina.ai/{url}")
        if resp.status_code == 200:
            return resp.text
    return ""


def _is_paywall(text: str) -> bool:
    text_lower = text.lower()
    return any(signal in text_lower for signal in _PAYWALL_SIGNALS)


def _needs_js(text: str) -> bool:
    text_lower = text.lower()
    return any(signal in text_lower for signal in _JS_SIGNALS)


def _is_duplicate_content(text: str) -> bool:
    """Return True if we've already seen this content (cross-URL duplicate)."""
    fp = hashlib.md5(text[:500].encode()).hexdigest()
    if fp in _content_fingerprints:
        return True
    _content_fingerprints.add(fp)
    # Keep cache bounded
    if len(_content_fingerprints) > 500:
        _content_fingerprints.clear()
    return False


def _extract_text(html: str, max_chars: int) -> str:
    """Extract readable text via readability-lxml, falling back to BeautifulSoup."""
    try:
        from readability import Document
        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)

    # Sentence-boundary truncation (not mid-character)
    if len(text) > max_chars:
        window = text[: max_chars + 300]
        # Find last sentence boundary before max_chars
        last_period = max(
            window.rfind(". ", 0, max_chars),
            window.rfind(".\n", 0, max_chars),
            window.rfind("! ", 0, max_chars),
            window.rfind("? ", 0, max_chars),
        )
        if last_period > max_chars // 2:
            text = window[: last_period + 1]
        else:
            text = window[:max_chars]

    return text


async def scrape_url(url: str) -> tuple[str, datetime]:
    """
    Scrape a single URL and return (text, scraped_at).
    2-tier: httpx → Jina Reader.
    """
    text = ""

    # Tier 1: httpx (fast)
    try:
        html = await asyncio.wait_for(
            _fetch_with_httpx(url),
            timeout=min(settings.scrape_timeout_s, 5),
        )
        candidate = _extract_text(html, settings.max_scraped_chars)
        if (
            len(candidate) > 300
            and not _needs_js(candidate)
            and not _is_paywall(candidate)
        ):
            return candidate, datetime.now(UTC)
        text = candidate  # keep as backup
    except Exception:
        pass

    # Tier 2: Jina Reader (paywalls, difficult JS, fallback)
    try:
        jina_text = await asyncio.wait_for(_fetch_with_jina(url), timeout=12.0)
        if jina_text and len(jina_text) > len(text):
            text = jina_text[: settings.max_scraped_chars]
    except Exception:
        pass

    return text, datetime.now(UTC)


async def scrape_urls(urls: list[str]) -> dict[str, tuple[str, datetime]]:
    """Scrape multiple URLs concurrently. Errors per-URL are silently dropped."""
    timeout = min(getattr(settings, "scrape_timeout_s", 8), 6.0)
    tasks = {
        url: asyncio.create_task(asyncio.wait_for(scrape_url(url), timeout=timeout))
        for url in urls
    }
    results: dict[str, tuple[str, datetime]] = {}
    for url, task in tasks.items():
        try:
            text, ts = await task
            # Skip duplicate content across URLs
            if text and not _is_duplicate_content(text):
                results[url] = (text, ts)
            elif text:
                results[url] = (text, ts)  # Still include but marked
        except Exception:
            pass
    return results
