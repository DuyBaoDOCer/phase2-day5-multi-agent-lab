"""Tracing hooks.

Supports two backends (automatically selected):
1. **File-based JSON tracer** (always active): writes span logs to ``reports/traces/``.
2. **LangSmith** (optional): activated when ``LANGSMITH_API_KEY`` is set in the environment.

Agents call ``trace_span(name, attributes)`` as a context manager. The span's duration
is measured and appended to an in-memory log. Call ``flush_trace_log(path)`` at the
end of a run to persist the log.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# Module-level in-memory span accumulator for the current run
_span_log: list[dict[str, Any]] = []


def setup_langsmith_tracing() -> bool:
    """Enable LangSmith tracing if the API key is configured.

    Returns True if LangSmith was activated, False otherwise.
    Sets environment variables that LangChain/LangGraph check at import time.
    """
    api_key = os.getenv("LANGSMITH_API_KEY") or ""
    project = os.getenv("LANGSMITH_PROJECT", "multi-agent-research-lab")
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    logger.info("LangSmith tracing enabled: project=%s", project)
    return True


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that measures a named span and stores it.

    Usage::

        with trace_span("researcher_search", {"query": q}) as span:
            results = search_client.search(q)
            span["results_count"] = len(results)
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "attributes": attributes or {},
        "duration_seconds": None,
    }
    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        _span_log.append(span)
        logger.debug("trace_span %s completed in %.3fs", name, span["duration_seconds"])


def flush_trace_log(output_path: Path | str | None = None) -> Path:
    """Persist the in-memory span log to a JSON file.

    Args:
        output_path: explicit file path; defaults to
            ``reports/traces/{timestamp}.json``.

    Returns the path where the trace was written.
    """
    if output_path is None:
        traces_dir = Path("reports/traces")
        traces_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = traces_dir / f"trace_{ts}.json"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_span_log, fh, indent=2, default=str)

    logger.info("Trace log flushed → %s (%d spans)", out, len(_span_log))
    return out


def reset_trace_log() -> None:
    """Clear the in-memory span log (useful between benchmark runs)."""
    _span_log.clear()
