"""Command-line entrypoint for the multi-agent research lab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import flush_trace_log, reset_trace_log, setup_langsmith_tracing

app = typer.Typer(help="Multi-Agent Research Lab CLI — Supervisor + Researcher + Analyst + Writer")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if setup_langsmith_tracing():
        console.print("[dim]LangSmith tracing active[/dim]")


def _single_agent_runner(query: str) -> ResearchState:
    """Run a single-agent baseline: one LLM call that does everything."""
    from multi_agent_research_lab.services import build_llm_client  # noqa: PLC0415

    llm = build_llm_client()
    system_prompt = (
        "You are a research assistant. Given a query, produce a comprehensive, well-structured "
        "response of approximately 500 words. Include facts, key findings, and a references section. "
        "Cite any sources you mention."
    )
    response = llm.complete(system_prompt, query)

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    state.final_answer = response.content
    state.agent_results = []  # no sub-agents in baseline
    return state


def _multi_agent_runner(query: str) -> ResearchState:
    """Run the full multi-agent workflow."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: Annotated[bool, typer.Option("--trace/--no-trace", help="Save JSON trace")] = True,
) -> None:
    """Run a real single-agent baseline (one Mistral call)."""
    _init()
    console.print(Panel.fit(f"[bold]Query:[/bold] {query}", title="Single-Agent Baseline"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        p.add_task("Calling Mistral AI...", total=None)
        reset_trace_log()
        try:
            state = _single_agent_runner(query)
        except Exception as exc:
            console.print(Panel.fit(str(exc), title="Error", style="red"))
            raise typer.Exit(code=1) from exc

    if save_trace:
        trace_path = flush_trace_log()
        console.print(f"[dim]Trace saved → {trace_path}[/dim]")

    console.print(Panel.fit(state.final_answer or "(no answer)", title="Final Answer", style="green"))
    if state.agent_results:
        total_cost = sum(r.metadata.get("cost_usd") or 0.0 for r in state.agent_results)
        console.print(f"[dim]Estimated cost: ${total_cost:.4f} USD[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: Annotated[bool, typer.Option("--trace/--no-trace", help="Save JSON trace")] = True,
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""
    _init()
    console.print(Panel.fit(f"[bold]Query:[/bold] {query}", title="Multi-Agent Workflow"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        p.add_task("Running agents...", total=None)
        reset_trace_log()
        try:
            state = _multi_agent_runner(query)
        except Exception as exc:
            console.print(Panel.fit(str(exc), title="Error", style="red"))
            raise typer.Exit(code=1) from exc

    if save_trace:
        trace_path = flush_trace_log()
        console.print(f"[dim]Trace saved → {trace_path}[/dim]")

    route_str = " → ".join(state.route_history) if state.route_history else "N/A"
    console.print(f"[dim]Route: START → {route_str}[/dim]")
    console.print(f"[dim]Sources: {len(state.sources)} | Iterations: {state.iteration}[/dim]")
    if state.errors:
        console.print(f"[yellow]Warnings: {state.errors}[/yellow]")

    console.print(Panel.fit(state.final_answer or "(no answer)", title="Final Answer", style="green"))
    total_cost = sum((r.metadata.get("cost_usd") or 0.0) for r in state.agent_results)
    console.print(f"[dim]Estimated total cost: ${total_cost:.4f} USD[/dim]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output report path")] = "reports/benchmark_report.md",
) -> None:
    """Run both baseline and multi-agent, compare, and save benchmark_report.md."""
    _init()
    console.print(Panel.fit(f"[bold]Benchmarking:[/bold] {query}", title="Benchmark"))

    all_metrics = []
    all_states = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        # Baseline run
        task = p.add_task("[cyan]Running single-agent baseline...", total=None)
        reset_trace_log()
        baseline_state, baseline_metrics = run_benchmark("single-agent-baseline", query, _single_agent_runner)
        flush_trace_log()
        all_metrics.append(baseline_metrics)
        all_states.append(baseline_state)
        p.remove_task(task)
        console.print(
            f"[green]✓[/green] Baseline done: {baseline_metrics.latency_seconds:.1f}s  "
            f"quality={baseline_metrics.quality_score}"
        )

        # Multi-agent run
        task = p.add_task("[cyan]Running multi-agent workflow...", total=None)
        reset_trace_log()
        multi_state, multi_metrics = run_benchmark("multi-agent-workflow", query, _multi_agent_runner)
        flush_trace_log()
        all_metrics.append(multi_metrics)
        all_states.append(multi_state)
        p.remove_task(task)
        console.print(
            f"[green]✓[/green] Multi-agent done: {multi_metrics.latency_seconds:.1f}s  "
            f"quality={multi_metrics.quality_score}"
        )

    # Render + save report
    report = render_markdown_report(all_metrics, all_states)
    report_path = Path(output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    console.print(f"\n[bold green]Benchmark report saved → {report_path}[/bold green]")
    console.print(report)


if __name__ == "__main__":
    app()
