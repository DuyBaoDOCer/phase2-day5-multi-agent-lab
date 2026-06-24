"""LangGraph multi-agent workflow.

Architecture:
  START → supervisor ──┬──► researcher → supervisor
                       ├──► analyst    → supervisor
                       ├──► writer     → supervisor
                       └──► END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services import build_llm_client, build_search_client

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    """Convert Pydantic state to dict for LangGraph node return."""
    return state.model_dump()


class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph.

    Keep orchestration here; keep agent internals in ``agents/``.
    """

    def __init__(self) -> None:
        llm = build_llm_client()
        search = build_search_client()

        self._supervisor = SupervisorAgent(llm_client=llm)
        self._researcher = ResearcherAgent(llm_client=llm, search_client=search)
        self._analyst = AnalystAgent(llm_client=llm)
        self._writer = WriterAgent(llm_client=llm)

    # ── Node functions ──────────────────────────────────────────────────────

    def _supervisor_node(self, state: dict[str, Any]) -> dict[str, Any]:
        s = ResearchState(**state)
        result = self._supervisor.run(s)
        return _state_to_dict(result)

    def _researcher_node(self, state: dict[str, Any]) -> dict[str, Any]:
        s = ResearchState(**state)
        result = self._researcher.run(s)
        return _state_to_dict(result)

    def _analyst_node(self, state: dict[str, Any]) -> dict[str, Any]:
        s = ResearchState(**state)
        result = self._analyst.run(s)
        return _state_to_dict(result)

    def _writer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        s = ResearchState(**state)
        result = self._writer.run(s)
        return _state_to_dict(result)

    # ── Conditional edge ────────────────────────────────────────────────────

    @staticmethod
    def _route(state: dict[str, Any]) -> str:
        """Read last entry in route_history to decide the next node."""
        history: list[str] = state.get("route_history", [])
        if not history:
            return END  # type: ignore[return-value]
        last = history[-1]
        if last == "done" or last not in {"researcher", "analyst", "writer"}:
            return END  # type: ignore[return-value]
        return last

    # ── Build & run ─────────────────────────────────────────────────────────

    def build(self) -> Any:
        """Create a compiled LangGraph graph.

        Nodes: supervisor, researcher, analyst, writer.
        Conditional edges from supervisor decide which worker runs next.
        Workers always return to supervisor for re-routing.
        Stop condition: route == 'done' or route not in valid set.
        """
        graph: StateGraph = StateGraph(dict)  # type: ignore[type-arg]

        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)

        # Entry point
        graph.add_edge(START, "supervisor")

        # Supervisor decides what runs next
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )

        # Workers always return to supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile and execute the graph, return final ResearchState.

        The graph runs until the supervisor routes to 'done' or
        max_iterations is reached (supervisor guardrail handles it).
        """
        logger.info("MultiAgentWorkflow.run: query=%r", state.request.query)
        compiled = self.build()

        initial_dict = state.model_dump()
        final_dict: dict[str, Any] = compiled.invoke(initial_dict)

        return ResearchState(**final_dict)
