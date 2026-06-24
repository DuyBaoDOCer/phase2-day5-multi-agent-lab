"""Analyst agent — turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a critical research analyst. Given research notes, produce a rigorous analysis.

Format your output as:

## Key Claims
- List the 3-5 most important claims found in the research notes.

## Evidence Assessment
- For each key claim, rate the evidence as Strong / Moderate / Weak and explain briefly.

## Competing Viewpoints
- Note if the sources agree or disagree on any point. Summarize each side.

## Analyst Verdict
- 2-3 sentence synthesis: what is the consensus? what is still uncertain?

Keep total length to 300-500 words. Be rigorous: do not accept weak evidence without flagging it.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``.

        Reads research_notes from state and uses LLM to extract key claims,
        compare viewpoints, and flag weak evidence.
        """
        from multi_agent_research_lab.observability.tracing import trace_span

        if not state.research_notes:
            logger.warning("AnalystAgent: no research notes to analyse")
            state.errors.append("analyst: research_notes is empty, skipping analysis")
            state.analysis_notes = "No research notes available for analysis."
            return state

        logger.info("AnalystAgent: analysing %d chars of research notes", len(state.research_notes))
        state.add_trace_event("analyst_start", {"notes_length": len(state.research_notes)})

        with trace_span("analyst_run", {"notes_length": len(state.research_notes)}) as span:
            user_prompt = (
                f"Research Query: {state.request.query}\n\n"
                f"Research Notes:\n{state.research_notes}\n\n"
                "Please produce a structured analysis."
            )

            llm_response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = llm_response.content

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=llm_response.content,
                    metadata={
                        "input_tokens": llm_response.input_tokens,
                        "output_tokens": llm_response.output_tokens,
                        "cost_usd": llm_response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "analyst_done",
                {
                    "analysis_length": len(state.analysis_notes),
                    "cost_usd": llm_response.cost_usd,
                },
            )
            span["attributes"].update({
                "analysis_length": len(state.analysis_notes),
                "cost_usd": llm_response.cost_usd,
            })
            logger.info("AnalystAgent: done, analysis=%d chars", len(state.analysis_notes))
            return state
