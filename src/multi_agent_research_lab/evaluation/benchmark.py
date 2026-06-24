"""Benchmark runner for single-agent vs multi-agent comparison.

Measures:
- Latency (wall-clock time)
- Estimated token cost (USD)
- Quality score heuristic (word count, citation coverage)
- Error count
"""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _compute_quality_score(state: ResearchState) -> float:
    """Heuristic quality score 0-10.

    Factors:
    - Final answer exists (0 or continue)
    - Word count >= 300 → up to 4 pts
    - Citation coverage (source titles or [bracket refs] in answer) → up to 4 pts
    - No errors → 2 pts
    """
    if not state.final_answer:
        return 0.0

    score = 0.0

    # Word count factor (max 4)
    word_count = len(state.final_answer.split())
    if word_count >= 400:
        score += 4.0
    elif word_count >= 200:
        score += 2.0 + 2.0 * (word_count - 200) / 200
    else:
        score += 2.0 * word_count / 200

    # Citation coverage (max 4)
    has_bracket_refs = bool(re.search(r"\[.{3,60}\]", state.final_answer))
    cited_source_count = sum(
        1 for doc in state.sources
        if doc.title.lower() in state.final_answer.lower()
    )
    if state.sources:
        citation_ratio = cited_source_count / len(state.sources)
        score += 4.0 * (0.5 * citation_ratio + 0.5 * (1.0 if has_bracket_refs else 0.0))
    else:
        score += 2.0 if has_bracket_refs else 0.0

    # Error penalty (max 2)
    if not state.errors:
        score += 2.0
    elif len(state.errors) == 1:
        score += 1.0

    return min(round(score, 1), 10.0)


def _total_cost(state: ResearchState) -> float | None:
    """Sum costs across all agent results."""
    costs = [
        r.metadata.get("cost_usd", 0.0) or 0.0
        for r in state.agent_results
        if isinstance(r.metadata.get("cost_usd"), (int, float))
    ]
    return sum(costs) if costs else None


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, token cost, quality, and return metrics.

    Args:
        run_name: Label for this run (e.g. "baseline" or "multi-agent").
        query: Research query string.
        runner: Callable that accepts a query str and returns ResearchState.

    Returns:
        (final_state, BenchmarkMetrics)
    """
    logger.info("Benchmark start: run_name=%r query=%r", run_name, query)
    started = perf_counter()

    try:
        state = runner(query)
        error_note = ""
    except Exception as exc:
        logger.error("Runner %r failed: %s", run_name, exc)
        # Return empty state for reporting
        from multi_agent_research_lab.core.schemas import ResearchQuery  # noqa: PLC0415
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        error_note = f"FAILED: {exc}"

    latency = perf_counter() - started
    quality = _compute_quality_score(state)
    cost = _total_cost(state)
    word_count = len(state.final_answer.split()) if state.final_answer else 0
    error_count = len(state.errors)

    notes_parts = [f"words={word_count}", f"errors={error_count}"]
    if error_note:
        notes_parts.append(error_note)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=", ".join(notes_parts),
    )

    logger.info(
        "Benchmark done: run=%r latency=%.2fs quality=%.1f cost=%s errors=%d",
        run_name, latency, quality, cost, error_count,
    )
    return state, metrics
