"""
AgentDBG CLI - Command line interface for the debugger.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="agentdbg")
def main() -> None:
    """AgentDBG - Chrome DevTools for AI Agents

    Real-time debugging for AI agent execution.
    """
    pass


@main.command()
@click.argument("script", type=click.Path(exists=True))
@click.argument("args", nargs=-1)
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8765, help="WebSocket server port")
@click.option("--ui-port", default=8766, help="UI server port")
@click.option("--no-ui", is_flag=True, help="Don't start the UI server")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
@click.option("--pause-on-start", is_flag=True, help="Pause execution at start")
@click.option("--pause-on-error", is_flag=True, default=True, help="Pause on errors")
@click.option("--cost-limit", type=float, help="Pause when cost exceeds limit")
def run(
    script: str,
    args: tuple[str, ...],
    host: str,
    port: int,
    ui_port: int,
    no_ui: bool,
    no_browser: bool,
    pause_on_start: bool,
    pause_on_error: bool,
    cost_limit: float | None,
) -> None:
    """Run a Python script with AgentDBG instrumentation.

    Example:
        agentdbg run my_agent.py --cost-limit 1.0
    """
    console.print(Panel.fit(
        "[bold blue]AgentDBG[/bold blue] - Real-time AI Agent Debugger",
        subtitle="Press Ctrl+C to stop",
    ))

    # Import here to avoid slow startup
    from agentdbg.config import DebugConfig
    from agentdbg.core import AgentDebugger, get_debugger
    from agentdbg.instrumentors import auto_instrument
    from agentdbg.server.websocket_server import start_server as start_ws_server
    from agentdbg.server.http_server import HTTPServer

    # Configure debugger
    config = DebugConfig(
        host=host,
        port=port,
        ui_port=ui_port,
        auto_pause_on_error=pause_on_error,
        auto_pause_on_cost=cost_limit,
    )

    # Initialize debugger with config
    debugger = get_debugger()
    debugger.config = config

    # Start servers
    console.print(f"[green]✓[/green] Starting WebSocket server on ws://{host}:{port}")
    ws_server = start_ws_server(host=host, port=port, background=True)

    if not no_ui:
        console.print(f"[green]✓[/green] Starting UI server on http://{host}:{ui_port}")
        http_server = HTTPServer(host=host, port=ui_port)
        http_server.start(background=True)

        if not no_browser:
            webbrowser.open(f"http://{host}:{ui_port}")

    # Auto-instrument LLM providers
    console.print("[green]✓[/green] Auto-instrumenting OpenAI, Anthropic, LangChain")
    auto_instrument()

    # Pause on start if requested
    if pause_on_start:
        console.print("[yellow]⏸[/yellow] Paused at start. Use UI to resume.")
        debugger.pause()

    console.print(f"[green]✓[/green] Running: {script}")
    console.print()

    # Run the script
    script_path = Path(script).resolve()
    script_dir = script_path.parent

    # Prepare sys.argv
    original_argv = sys.argv.copy()
    sys.argv = [str(script_path)] + list(args)

    # Add script directory to path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        # Execute the script
        exec(compile(script_path.read_text(), script_path, "exec"), {"__name__": "__main__"})

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if debugger.is_paused:
            console.print("[yellow]Debugger is paused. Use UI to inspect state.[/yellow]")
            # Keep running to allow inspection
            try:
                input("Press Enter to exit...")
            except KeyboardInterrupt:
                pass
    finally:
        sys.argv = original_argv

    # Print summary
    _print_summary(debugger)


@main.command()
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8765, help="WebSocket server port")
@click.option("--ui-port", default=8766, help="UI server port")
def server(host: str, port: int, ui_port: int) -> None:
    """Start the AgentDBG server without running a script.

    Useful for connecting to from external applications.
    """
    from agentdbg.server.websocket_server import start_server as start_ws_server
    from agentdbg.server.http_server import HTTPServer

    console.print(Panel.fit(
        "[bold blue]AgentDBG Server[/bold blue]",
        subtitle="Press Ctrl+C to stop",
    ))

    console.print(f"[green]✓[/green] WebSocket: ws://{host}:{port}")
    console.print(f"[green]✓[/green] UI: http://{host}:{ui_port}")

    ws_server = start_ws_server(host=host, port=port, background=True)
    http_server = HTTPServer(host=host, port=ui_port)

    webbrowser.open(f"http://{host}:{ui_port}")

    try:
        http_server.start(background=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")


@main.command()
@click.option("--db-path", default=".agentdbg/traces.db", help="Database path")
def traces(db_path: str) -> None:
    """List recent traces from storage."""
    from agentdbg.storage import SQLiteStorage

    storage = SQLiteStorage(db_path)
    recent = storage.get_traces(limit=20)

    if not recent:
        console.print("[yellow]No traces found[/yellow]")
        return

    table = Table(title="Recent Traces")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Cost", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Duration", justify="right")

    for trace in recent:
        status_style = "green" if trace.status.value == "completed" else "red"
        duration = ""
        if trace.end_time and trace.start_time:
            duration = f"{(trace.end_time - trace.start_time) * 1000:.0f}ms"

        table.add_row(
            trace.trace_id[:8],
            trace.name[:30] if trace.name else "-",
            f"[{status_style}]{trace.status.value}[/{status_style}]",
            f"${trace.total_cost:.4f}",
            str(trace.total_tokens),
            duration,
        )

    console.print(table)


@main.command()
@click.option("--db-path", default=".agentdbg/traces.db", help="Database path")
def stats(db_path: str) -> None:
    """Show storage statistics."""
    from agentdbg.storage import SQLiteStorage

    storage = SQLiteStorage(db_path)
    s = storage.get_stats()

    table = Table(title="AgentDBG Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Traces", str(s["trace_count"]))
    table.add_row("Total Spans", str(s["span_count"]))
    table.add_row("Total Cost", f"${s['total_cost']:.4f}")
    table.add_row("Total Tokens", f"{s['total_tokens']:,}")
    table.add_row("Error Count", str(s["error_count"]))
    table.add_row("Error Rate", f"{s['error_rate']*100:.1f}%")

    console.print(table)


@main.command()
@click.option("--db-path", default=".agentdbg/traces.db", help="Database path")
@click.option("--days", default=7, help="Delete traces older than N days")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def cleanup(db_path: str, days: int, yes: bool) -> None:
    """Clean up old traces from storage."""
    from agentdbg.storage import SQLiteStorage

    storage = SQLiteStorage(db_path)

    if not yes:
        if not click.confirm(f"Delete traces older than {days} days?"):
            return

    count = storage.delete_old_traces(days)
    console.print(f"[green]✓[/green] Deleted {count} traces")


def _print_summary(debugger: "AgentDebugger") -> None:
    """Print execution summary."""
    traces = debugger.get_all_traces()

    if not traces:
        return

    console.print()
    console.print(Panel.fit("[bold]Execution Summary[/bold]"))

    total_cost = sum(t.total_cost for t in traces)
    total_tokens = sum(t.total_tokens for t in traces)
    total_spans = sum(len(t.spans) for t in traces)
    errors = sum(1 for t in traces if t.status.value == "error")

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Traces", str(len(traces)))
    table.add_row("Spans", str(total_spans))
    table.add_row("Total Cost", f"${total_cost:.4f}")
    table.add_row("Total Tokens", f"{total_tokens:,}")
    table.add_row("Errors", f"[{'red' if errors else 'green'}]{errors}[/]")

    console.print(table)


if __name__ == "__main__":
    main()
