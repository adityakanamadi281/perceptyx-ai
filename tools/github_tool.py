"""
tools/github_tool.py
--------------------
GitHub data retrieval via the REST API.
Supports: repo info, commits, PRs, and README excerpt.

Usage: parse a sub-query like "owner/repo commits" or "owner/repo PRs"
to extract the repo slug, then fetch the relevant data.
"""

from __future__ import annotations

import re
import time

import httpx
import structlog

from config.settings import settings
from models.schemas import GitHubCommit, GitHubOutput, GitHubPR

log = structlog.get_logger()

_HEADERS_BASE = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _auth_headers() -> dict[str, str]:
    h = dict(_HEADERS_BASE)
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def extract_repo_slug(text: str) -> str | None:
    """Extract an owner/repo slug from free text. Returns None if not found."""
    match = re.search(r"([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)", text)
    return match.group(1) if match else None


async def fetch_github_data(repo: str, sub_query: str) -> GitHubOutput:
    """
    Fetch commits, PRs, and README for a given owner/repo slug.
    Gracefully returns partial data on errors.
    """
    t0 = time.perf_counter()
    base = settings.github_api_endpoint
    headers = _auth_headers()
    commits: list[GitHubCommit] = []
    prs: list[GitHubPR] = []
    readme: str | None = None

    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:

        # ── Commits ────────────────────────────────────────────────────────────
        try:
            r = await client.get(
                f"{base}/repos/{repo}/commits",
                params={"per_page": settings.github_max_commits},
            )
            if r.status_code == 200:
                for c in r.json():
                    commits.append(GitHubCommit(
                        sha=c["sha"][:7],
                        message=c["commit"]["message"].split("\n")[0][:120],
                        author=c["commit"]["author"]["name"],
                        date=c["commit"]["author"]["date"],
                        url=c["html_url"],
                    ))
        except Exception as exc:
            log.warning("github_commits_failed", repo=repo, error=str(exc))

        # ── Pull Requests ──────────────────────────────────────────────────────
        try:
            r = await client.get(
                f"{base}/repos/{repo}/pulls",
                params={"per_page": settings.github_max_prs, "state": "all"},
            )
            if r.status_code == 200:
                for p in r.json():
                    prs.append(GitHubPR(
                        number=p["number"],
                        title=p["title"],
                        state=p["state"],
                        body=(p.get("body") or "")[:400],
                        author=p["user"]["login"],
                        created_at=p["created_at"],
                        url=p["html_url"],
                    ))
        except Exception as exc:
            log.warning("github_prs_failed", repo=repo, error=str(exc))

        # ── README ─────────────────────────────────────────────────────────────
        try:
            import base64
            r = await client.get(f"{base}/repos/{repo}/readme")
            if r.status_code == 200:
                content_b64 = r.json().get("content", "")
                readme = base64.b64decode(content_b64).decode("utf-8", errors="ignore")[:2000]
        except Exception as exc:
            log.warning("github_readme_failed", repo=repo, error=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("github_done", repo=repo, commits=len(commits), prs=len(prs),
             latency_ms=round(latency_ms, 1))

    return GitHubOutput(
        repo=repo,
        commits=commits,
        pull_requests=prs,
        readme_excerpt=readme,
        latency_ms=latency_ms,
    )
