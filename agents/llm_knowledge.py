"""
agents/llm_knowledge.py
-----------------------
New agent that prompts the LLM to extract its own parametric knowledge
as structured JSON facts, and runs search and LLM extraction in parallel.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import structlog

from config.settings import settings
from core.observability import TelemetryCallback
from models.schemas import PipelineTrace, SearchResult

log = structlog.get_logger()

_LLM_KNOWLEDGE_SYSTEM = """\
You are a factual knowledge extraction system. Based ONLY on your parametric knowledge (what you know without web search), extract key facts relevant to the query.
Provide the facts as a structured JSON list of facts, each with a confidence score between 0.0 and 1.0.
Respond ONLY with a JSON object of this structure:
{
  "facts": [
    {"fact": "fact string here", "confidence": 0.95},
    {"fact": "another fact here", "confidence": 0.8}
  ]
}
Do not add any preamble, markdown code blocks, or extra text.
"""


async def _persist_semantic_facts(facts: list[dict[str, Any]]) -> None:
    try:
        from rag.vectorstore import upsert_document

        for f in facts:
            fact_text = f.get("fact")
            confidence = f.get("confidence", 1.0)
            if not fact_text:
                continue

            h = hashlib.sha256(fact_text.encode("utf-8")).hexdigest()
            uid = f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

            await upsert_document(
                collection_name="semantic_knowledge",
                text=fact_text,
                metadata={"confidence": confidence},
                id=uid,
            )
            log.info("semantic_fact_persisted", fact=fact_text[:50], confidence=confidence)
    except Exception as e:
        log.warning("persist_semantic_facts_failed", error=str(e))


async def run_llm_knowledge_agent(query: str, trace: PipelineTrace) -> list[dict[str, Any]]:
    """
    Prompts the LLM to extract its own parametric knowledge as structured JSON facts.
    Capped at a hard timeout of 6 seconds.
    """
    from providers.llm import llm_invoke

    callback = TelemetryCallback("llm_knowledge_agent", trace)

    try:
        raw = await asyncio.wait_for(
            llm_invoke(
                system=_LLM_KNOWLEDGE_SYSTEM,
                user=f"Query: {query}",
                callback=callback,
            ),
            timeout=6.0,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        facts = data.get("facts", [])

        # Persist high-confidence facts (confidence >= 0.8) to semantic memory as background task
        high_conf_facts = [f for f in facts if f.get("confidence", 0.0) >= 0.8]
        if high_conf_facts and settings.enable_self_learning:
            asyncio.create_task(_persist_semantic_facts(high_conf_facts))

        return facts
    except Exception as exc:
        log.warning("llm_knowledge_agent_failed", error=str(exc))
        return []


async def run_parallel_search_and_llm(
    query: str,
    trace: PipelineTrace,
) -> tuple[list[SearchResult], list[dict[str, Any]]]:
    """
    Parallel entrypoint: launches web search and LLM knowledge extraction concurrently.
    Used by agents/search.py and the orchestrator's fast path.
    """
    from tools.duckduckgo import ddg_search
    from tools.search_aggregator import multi_provider_search

    async def run_web_search():
        try:
            results = await multi_provider_search(query, n=settings.max_search_results)
            if results:
                return results
        except Exception as e:
            log.warning("parallel_search_aggregator_failed", error=str(e))

        try:
            return await ddg_search(query, num=settings.max_search_results)
        except Exception as e:
            log.warning("parallel_search_ddg_failed", error=str(e))
            return []

    web_results, facts = await asyncio.gather(
        run_web_search(),
        run_llm_knowledge_agent(query, trace),
    )
    return web_results, facts
