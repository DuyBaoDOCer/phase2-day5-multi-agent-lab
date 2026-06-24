"""Tests for implemented agents (no longer student TODOs)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, SupervisorAgent, WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def _make_state(**kwargs: object) -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), **kwargs)  # type: ignore[arg-type]


def _mock_llm(content: str = "Mock LLM response") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(content=content, input_tokens=50, output_tokens=80, cost_usd=0.0001)
    return llm


# ── SupervisorAgent ──────────────────────────────────────────────────────────

def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    state = _make_state()
    agent = SupervisorAgent(llm_client=None)
    result = agent.run(state)
    assert result.route_history[-1] == "researcher"


def test_supervisor_routes_to_analyst_when_has_research_notes() -> None:
    state = _make_state(research_notes="Some research notes")
    agent = SupervisorAgent(llm_client=None)
    result = agent.run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_when_has_analysis() -> None:
    state = _make_state(research_notes="Notes", analysis_notes="Analysis")
    agent = SupervisorAgent(llm_client=None)
    result = agent.run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_done_when_final_answer_exists() -> None:
    state = _make_state(research_notes="Notes", analysis_notes="Analysis", final_answer="Done!")
    agent = SupervisorAgent(llm_client=None)
    result = agent.run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_enforces_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supervisor must stop if iteration >= max_iterations."""
    monkeypatch.setenv("MAX_ITERATIONS", "2")
    # Clear settings cache so new env var is picked up
    from multi_agent_research_lab.core.config import get_settings
    get_settings.cache_clear()

    state = _make_state()
    state.iteration = 2  # already at limit
    agent = SupervisorAgent(llm_client=None)
    result = agent.run(state)
    assert result.route_history[-1] == "done"
    get_settings.cache_clear()


# ── ResearcherAgent ──────────────────────────────────────────────────────────

def test_researcher_populates_research_notes() -> None:
    llm = _mock_llm("## Key Findings\n- GraphRAG is cool")
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SourceDocument(title="GraphRAG Paper", url="https://example.com", snippet="snippet")
    ]
    state = _make_state()
    agent = ResearcherAgent(llm_client=llm, search_client=mock_search)
    result = agent.run(state)

    assert result.research_notes is not None
    assert len(result.sources) == 1
    assert result.sources[0].title == "GraphRAG Paper"


def test_researcher_handles_empty_search_results() -> None:
    llm = _mock_llm()
    mock_search = MagicMock()
    mock_search.search.return_value = []
    state = _make_state()
    agent = ResearcherAgent(llm_client=llm, search_client=mock_search)
    result = agent.run(state)
    assert result.research_notes is not None
    assert len(result.errors) > 0


# ── AnalystAgent ──────────────────────────────────────────────────────────────

def test_analyst_populates_analysis_notes() -> None:
    llm = _mock_llm("## Key Claims\n- Claim 1")
    state = _make_state(research_notes="Research notes content")
    agent = AnalystAgent(llm_client=llm)
    result = agent.run(state)
    assert result.analysis_notes is not None


def test_analyst_skips_when_no_research_notes() -> None:
    llm = _mock_llm()
    state = _make_state()
    agent = AnalystAgent(llm_client=llm)
    result = agent.run(state)
    assert result.analysis_notes is not None
    assert len(result.errors) > 0
    llm.complete.assert_not_called()


# ── WriterAgent ───────────────────────────────────────────────────────────────

def test_writer_populates_final_answer() -> None:
    llm = _mock_llm("## Summary\nThis is the final answer with [Source] citation.")
    state = _make_state(research_notes="Notes", analysis_notes="Analysis")
    agent = WriterAgent(llm_client=llm)
    result = agent.run(state)
    assert result.final_answer is not None
    assert len(result.final_answer) > 0
