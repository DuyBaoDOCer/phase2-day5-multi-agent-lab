"""Critic agent — validates final answer quality and citation coverage."""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a quality-control critic for AI-generated research reports. Review the final answer
and check for:

1. Citation coverage: Does each major claim have a source reference?
2. Factual consistency: Do the claims match the research notes (no hallucinations)?
3. Completeness: Does the answer fully address the query?
4. Clarity: Is the writing clear and well-structured?

Respond with a structured critique:

## Citation Coverage
- Score (0-10) and brief justification.

## Factual Consistency
- Score (0-10): flag any claim that seems unsupported or contradicts the notes.

## Completeness
- Score (0-10) and what (if anything) is missing.

## Overall Quality Score
- Single number 0-10 and 1-2 sentence summary.
"""


class CriticAgent(BaseAgent):
    """Optional fact-checking and quality-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final_answer and append quality findings to trace.

        Computes a citation_coverage_ratio from the final_answer text
        (fraction of source titles that appear in the answer) and asks
        LLM for a full quality critique.
        """
        from multi_agent_research_lab.observability.tracing import trace_span

        if not state.final_answer:
            logger.warning("CriticAgent: no final_answer to review")
            state.errors.append("critic: final_answer is empty")
            return state

        logger.info("CriticAgent: reviewing final answer")
        state.add_trace_event("critic_start", {})

        with trace_span("critic_run", {}) as span:
            # Quick heuristic: how many source titles appear in the final answer?
            cited_count = sum(
                1 for doc in state.sources
                if doc.title.lower() in state.final_answer.lower()
                or bool(re.search(r"\[.{3,50}\]", state.final_answer))
            )
            citation_ratio = cited_count / len(state.sources) if state.sources else 0.0

            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
                f"Final Answer:\n{state.final_answer}\n\n"
                f"Available Sources: {[doc.title for doc in state.sources]}\n\n"
                "Please critique the final answer."
            )

            llm_response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=llm_response.content,
                    metadata={
                        "citation_coverage_ratio": citation_ratio,
                        "input_tokens": llm_response.input_tokens,
                        "output_tokens": llm_response.output_tokens,
                        "cost_usd": llm_response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "critic_done",
                {
                    "citation_coverage_ratio": citation_ratio,
                    "critique_length": len(llm_response.content),
                },
            )
            span["attributes"].update({
                "citation_coverage_ratio": citation_ratio,
                "critique_length": len(llm_response.content),
            })
            logger.info("CriticAgent: citation_coverage_ratio=%.2f", citation_ratio)
            return state
