"""
core/router.py
--------------
Classifies incoming queries to determine if they need real-time web search
or can be answered directly using native LLM knowledge.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from core.observability import TelemetryCallback, get_logger
from models.schemas import PipelineTrace
from providers.gemini import get_gemini_llm

_SYSTEM = """\
You are an advanced query router. Analyze the user's query and determine if it requires real-time web search or fact-checking.

Search is REQUIRED for:
- Current events, news, or recent developments (e.g., "what happened in X yesterday").
- Real-time/time-sensitive data (e.g., weather, stock prices, current office holders, sports scores).
- Specific facts, figures, or claims that need verification/fact-checking.
- Product reviews, comparisons, or local business info.

Search is NOT REQUIRED (use native LLM knowledge) for:
- Coding, programming help, debugging, or script generation.
- Creative writing, brainstorming, translations, or text editing.
- General static knowledge (e.g., "how does photosynthesis work", "explain relativity", historical events, mathematical proofs).
- Conversational chat, advice, or opinion prompts.

Respond ONLY with a JSON object matching this schema (no code fences, no extra text):
{
  "requires_search": true,
  "reason": "Brief explanation of why search is or is not required"
}
"""


async def classify_query(query: str, trace: PipelineTrace) -> bool:
    """
    Classify whether a query needs web search or can be answered directly.
    """
    logger = get_logger("router", trace.run_id)
    callback = TelemetryCallback("router", trace)
    llm = get_gemini_llm()

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=query),
    ]

    logger.info("routing_classification_start", query=query)
    try:
        response = await llm.ainvoke(messages, config={"callbacks": [callback]})
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        requires_search = bool(data.get("requires_search", True))
        logger.info(
            "routing_classification_done",
            requires_search=requires_search,
            reason=data.get("reason", ""),
        )
        return requires_search
    except Exception as exc:
        logger.warning("routing_classification_failed", error=str(exc), fallback=True)
        return True
