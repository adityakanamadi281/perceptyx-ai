"""
providers/gemini.py
-------------------
Thin wrapper around google-generativeai via langchain-google-genai.
Returns a configured ChatGoogleGenerativeAI instance ready to receive
LangChain callbacks for telemetry.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings


@lru_cache(maxsize=1)
def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Returns a cached ChatGoogleGenerativeAI instance.
    Callbacks are attached per-call (not here) so each agent
    can inject its own TelemetryCallback.
    """
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_tokens,
        max_retries=2,
        # Disable safety filters for research context
        convert_system_message_to_human=True,
    )
