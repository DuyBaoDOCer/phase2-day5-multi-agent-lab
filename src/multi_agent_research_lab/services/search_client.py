"""Search client — Tavily implementation.

Bridges the Tavily API to the internal SourceDocument schema.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Tavily-powered search client.

    Falls back gracefully: if the query yields no results, returns an empty list
    rather than raising, so the ResearcherAgent can decide what to do next.
    """

    def __init__(self, api_key: str) -> None:
        from tavily import TavilyClient  # lazy import

        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search Tavily and return structured source documents.

        Each SourceDocument captures title, url, and a snippet (content field from Tavily).
        Score is stored in metadata for downstream quality filtering.
        """
        logger.info("SearchClient.search: query=%r max_results=%d", query, max_results)
        try:
            response = self._client.search(query=query, max_results=max_results, include_answer=False)
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            return []

        results: list[SourceDocument] = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Unknown"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score", 0.0)},
                )
            )

        logger.info("SearchClient found %d results", len(results))
        return results
