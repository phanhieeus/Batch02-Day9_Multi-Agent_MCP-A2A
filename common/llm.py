"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("CKEY_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("CKEY_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        streaming=True,
    )