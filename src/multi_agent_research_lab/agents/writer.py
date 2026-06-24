"""Writer agent — produces a polished final answer from research and analysis notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a skilled technical writer. Given research notes and an analysis, produce a polished,
well-structured response targeted at technical learners.

Requirements:
- Length: approximately 500 words.
- Structure: use clear headings (##) for major sections.
- Citations: reference source titles inline using [Source Title] notation.
- Tone: clear, authoritative, and accessible to a technical audience.
- End with a "## References" section listing all cited sources with their URLs.
- Do NOT hallucinate sources — only cite what was provided.
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer``.

        Synthesizes a clear, ~500-word response with citations from
        research_notes + analysis_notes + original sources list.
        """
        from multi_agent_research_lab.observability.tracing import trace_span

        if not state.research_notes and not state.analysis_notes:
            logger.warning("WriterAgent: no notes to write from")
            state.final_answer = "Insufficient research data to generate a response."
            state.errors.append("writer: both research and analysis notes are empty")
            return state

        logger.info("WriterAgent: composing final answer")
        state.add_trace_event("writer_start", {"query": state.request.query})

        with trace_span("writer_run", {"query": state.request.query}) as span:
            # Build reference list for the LLM from stored sources
            ref_lines = []
            for i, doc in enumerate(state.sources, start=1):
                ref_lines.append(f"[Ref {i}] {doc.title} – {doc.url or 'no URL'}")
            refs_block = "\n".join(ref_lines) if ref_lines else "No sources available."

            user_prompt = (
                f"Research Query: {state.request.query}\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
                f"Analysis Notes:\n{state.analysis_notes or 'N/A'}\n\n"
                f"Available Sources:\n{refs_block}\n\n"
                "Please write the final response (~500 words)."
            )

            llm_response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.final_answer = llm_response.content

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=llm_response.content,
                    metadata={
                        "input_tokens": llm_response.input_tokens,
                        "output_tokens": llm_response.output_tokens,
                        "cost_usd": llm_response.cost_usd,
                        "word_count": len(llm_response.content.split()),
                    },
                )
            )
            state.add_trace_event(
                "writer_done",
                {
                    "answer_length": len(state.final_answer),
                    "word_count": len(state.final_answer.split()),
                    "cost_usd": llm_response.cost_usd,
                },
            )
            span["attributes"].update({
                "answer_length": len(state.final_answer),
                "word_count": len(state.final_answer.split()),
                "cost_usd": llm_response.cost_usd,
            })
            logger.info(
                "WriterAgent: done, answer=%d words",
                len(state.final_answer.split()),
            )
            return state
