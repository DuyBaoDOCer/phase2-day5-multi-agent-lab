"""Supervisor / router — decides which worker runs next and when to stop."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research supervisor. Given the current research state, decide which agent should run next.

Available agents and when to use them:
- researcher : when we have no research notes yet, or when the notes are too thin.
- analyst    : when we have research notes but no analysis yet.
- writer     : when we have both research notes and analysis notes but no final answer.
- done       : when a satisfactory final answer exists.

Reply with EXACTLY one word: researcher, analyst, writer, or done.
Do NOT explain. Do NOT add punctuation.
"""


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Update ``state.route_history`` with the next route.

        Routing policy (with guardrails):
        1. Hard guardrail: if iteration >= max_iterations → done.
        2. Rule-based fast path (no LLM call needed for clear-cut cases).
        3. LLM-based routing for ambiguous cases (requires api key).
        """
        from multi_agent_research_lab.observability.tracing import trace_span

        with trace_span("supervisor_route") as span:
            settings = get_settings()

            # ── Guardrail: stop if iteration limit reached ──────────────────────
            if state.iteration >= settings.max_iterations:
                logger.warning("Supervisor: max_iterations=%d reached, forcing done", settings.max_iterations)
                state.record_route("done")
                state.add_trace_event("supervisor_route", {"next": "done", "reason": "max_iterations"})
                span["attributes"] = {"next": "done", "reason": "max_iterations"}
                return state

            # ── Rule-based fast path ────────────────────────────────────────────
            if not state.research_notes:
                next_agent = "researcher"
                reason = "no research notes yet"
            elif not state.analysis_notes:
                next_agent = "analyst"
                reason = "have notes, need analysis"
            elif not state.final_answer:
                next_agent = "writer"
                reason = "have analysis, need final answer"
            else:
                next_agent = "done"
                reason = "final answer exists"

            # ── Optional LLM-based override for complex situations ──────────────
            if self._llm and next_agent not in ("done",) and state.iteration > 0:
                user_prompt = (
                    f"Query: {state.request.query}\n"
                    f"Iteration: {state.iteration}\n"
                    f"Research notes present: {bool(state.research_notes)}\n"
                    f"Analysis notes present: {bool(state.analysis_notes)}\n"
                    f"Final answer present: {bool(state.final_answer)}\n"
                    f"Errors so far: {state.errors}\n"
                    f"Previous route history: {state.route_history}\n"
                    "Which agent should run next?"
                )
                try:
                    response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                    llm_choice = response.content.strip().lower().split()[0]
                    if llm_choice in {"researcher", "analyst", "writer", "done"}:
                        next_agent = llm_choice
                        reason = f"LLM decided ({response.input_tokens} tokens)"
                except Exception as exc:
                    logger.warning("Supervisor LLM routing failed, using rule-based: %s", exc)

            logger.info("Supervisor → %s (reason: %s)", next_agent, reason)
            state.record_route(next_agent)
            state.add_trace_event("supervisor_route", {"next": next_agent, "reason": reason})
            span["attributes"] = {"next": next_agent, "reason": reason}
            return state
