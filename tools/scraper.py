"""
tools/scraper.py
----------------
Async web scraper that:
1. Fetches the page with Playwright (handles JS-rendered content).
2. Falls back to httpx if Playwright is unavailable.
3. Strips HTML with readability-lxml to extract the main article text.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from config.settings import settings


def _run_playwright_in_thread(url: str) -> str:
    import sys
    if sys.platform == 'win32':
        # On Windows, SelectorEventLoop does not support subprocesses (which Playwright uses to launch the browser).
        # We create a ProactorEventLoop specifically for Playwright inside this background thread.
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _fetch():
        from playwright.async_api import async_playwright  # lazy import
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=settings.scrape_timeout_s * 1000)
                return await page.content()
            finally:
                await browser.close()
    try:
        return loop.run_until_complete(_fetch())
    finally:
        loop.close()


async def _fetch_with_playwright(url: str) -> str:
    """Use Playwright async API to render and return page HTML in a separate thread (for Windows event loop compatibility)."""
    return await asyncio.to_thread(_run_playwright_in_thread, url)



async def _fetch_with_httpx(url: str) -> str:
    """Lightweight fallback fetcher."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PerplexityAgent/0.1; +https://github.com/example)"
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


def _extract_text(html: str, max_chars: int) -> str:
    """Extract readable text via readability-lxml, falling back to BeautifulSoup."""
    try:
        from readability import Document  # readability-lxml

        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        # Remove scripts / styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    # Collapse blank lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)[:max_chars]


async def scrape_url(url: str) -> tuple[str, datetime]:
    """
    Scrape a single URL and return (text, scraped_at).
    Tries httpx first (fast/lightweight), falls back to Playwright for JS-heavy sites.
    """
    text = ""
    try:
        # 1. Try lightweight HTTPX first with a short timeout
        html = await asyncio.wait_for(
            _fetch_with_httpx(url),
            timeout=min(settings.scrape_timeout_s, 5),
        )
        text = _extract_text(html, settings.max_scraped_chars)
        # If we got substantial text and it doesn't indicate JS requirement, return it
        if len(text) > 300 and not any(kw in text.lower() for kw in ["enable javascript", "javascript is required", "javascript enabled"]):
            return text, datetime.now(timezone.utc)
    except Exception:
        pass

    try:
        # 2. Fall back to Playwright if HTTPX failed or page requires JS rendering
        html = await asyncio.wait_for(
            _fetch_with_playwright(url),
            timeout=settings.scrape_timeout_s,
        )
        text = _extract_text(html, settings.max_scraped_chars)
    except Exception:
        # If Playwright fails too, we keep the HTTPX result if we had one
        pass

    return text, datetime.now(timezone.utc)


async def scrape_urls(urls: list[str]) -> dict[str, tuple[str, datetime]]:
    """Scrape multiple URLs concurrently. Errors per-URL are silently dropped."""
    tasks = {url: asyncio.create_task(scrape_url(url)) for url in urls}
    results: dict[str, tuple[str, datetime]] = {}
    for url, task in tasks.items():
        try:
            results[url] = await task
        except Exception:
            pass
    return results
