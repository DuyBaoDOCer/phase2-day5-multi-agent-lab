"""Researcher agent — searches for sources and synthesises research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a meticulous research assistant. Given search results about a topic, synthesize them into
concise, well-structured research notes. Focus on factual accuracy.

Format:
## Key Findings
- Bullet points of the most important facts.

## Sources Summary
- Brief description of each source's contribution.

## Gaps & Uncertainties
- Note anything the sources do NOT cover well.

Keep the total length to 400-600 words. Cite sources by their title when referencing them.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm_client: LLMClient, search_client: SearchClient) -> None:
        self._llm = llm_client
        self._search = search_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``.

        Steps:
        1. Search Tavily for documents relevant to the query.
        2. Compile source snippets into a context block.
        3. Ask LLM to synthesize structured research notes.
        4. Record result in AgentResult and trace.
        """
        from multi_agent_research_lab.observability.tracing import trace_span
        
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info("ResearcherAgent: searching for %r (max=%d)", query, max_sources)
        state.add_trace_event("researcher_start", {"query": query})

        with trace_span("researcher_run", {"query": query}) as span:
            # 1. Fetch sources
            sources = self._search.search(query, max_results=max_sources)
            if not sources:
                msg = "No search results returned for query. Continuing with empty notes."
                logger.warning(msg)
                state.errors.append(f"researcher: {msg}")
                state.research_notes = f"No sources found for: {query}"
                span["attributes"]["error"] = msg
                return state

            state.sources = sources

            # 2. Build context block for LLM
            context_parts = []
            for i, doc in enumerate(sources, start=1):
                context_parts.append(
                    f"[Source {i}] Title: {doc.title}\nURL: {doc.url or 'N/A'}\nSnippet: {doc.snippet}"
                )
            context = "\n\n".join(context_parts)

            user_prompt = f"Topic: {query}\n\nSearch Results:\n{context}\n\nPlease write research notes."

            # 3. LLM synthesis
            logger.info("ResearcherAgent: calling LLM to synthesize notes")
            llm_response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.research_notes = llm_response.content

            # 4. Record result
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=llm_response.content,
                    metadata={
                        "sources_count": len(sources),
                        "input_tokens": llm_response.input_tokens,
                        "output_tokens": llm_response.output_tokens,
                        "cost_usd": llm_response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "researcher_done",
                {
                    "sources_count": len(sources),
                    "notes_length": len(state.research_notes),
                    "cost_usd": llm_response.cost_usd,
                },
            )
            span["attributes"].update({
                "sources_count": len(sources),
                "notes_length": len(state.research_notes),
                "cost_usd": llm_response.cost_usd,
            })
            logger.info("ResearcherAgent: done, notes=%d chars", len(state.research_notes))
            return state
