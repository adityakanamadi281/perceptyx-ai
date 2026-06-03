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
import sys
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from config.settings import settings


async def _fetch_with_playwright(url: str, timeout_s: float) -> str:
    """Use Playwright async API to render and return page HTML."""
    from playwright.async_api import async_playwright  # lazy import

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=int(timeout_s * 1000))
            html = await page.content()
        finally:
            await browser.close()
    return html


def _run_playwright_in_thread(url: str, timeout_s: float) -> str:
    """Helper run in a separate thread to ensure Proactor loop is used on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_fetch_with_playwright(url, timeout_s))
    finally:
        loop.close()


async def _fetch_with_httpx(url: str) -> str:
    """Lightweight fallback fetcher."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (compatible; PerplexityAgent/0.1; +https://github.com/example)")
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
    Tries Playwright first (in a separate thread to support SelectorEventLoop on Windows),
    falls back to httpx.
    """
    try:
        html = await asyncio.to_thread(_run_playwright_in_thread, url, settings.scrape_timeout_s)
    except Exception:
        html = await asyncio.wait_for(
            _fetch_with_httpx(url),
            timeout=settings.scrape_timeout_s,
        )

    text = _extract_text(html, settings.max_scraped_chars)
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
