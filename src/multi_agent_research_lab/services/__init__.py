"""Services package — factory helpers."""

from __future__ import annotations

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient

__all__ = ["LLMClient", "LLMResponse", "SearchClient", "build_llm_client", "build_search_client"]


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Construct an LLMClient from Settings. Raises if API key is missing."""
    cfg = settings or get_settings()
    if not cfg.mistral_api_key:
        raise AgentExecutionError(
            "MISTRAL_API_KEY is not set. Add it to your .env file."
        )
    return LLMClient(api_key=cfg.mistral_api_key, model=cfg.mistral_model)


def build_search_client(settings: Settings | None = None) -> SearchClient:
    """Construct a SearchClient from Settings. Raises if API key is missing."""
    cfg = settings or get_settings()
    if not cfg.tavily_api_key:
        raise AgentExecutionError(
            "TAVILY_API_KEY is not set. Add it to your .env file."
        )
    return SearchClient(api_key=cfg.tavily_api_key)
