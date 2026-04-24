"""CLI entrypoint using Typer.

Typer builds on Click but uses Python type hints for argument/option
definitions — less boilerplate, better IDE support. The `rich` integration
gives us colored output and markdown rendering for free.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from codebase_assistant.agents.base import ReactAgent
from codebase_assistant.models.base import Message
from codebase_assistant.models.factory import create_provider
from codebase_assistant.prompts import AGENT_SYSTEM_PROMPT, SYSTEM_PROMPT
from codebase_assistant.tools.registry import build_default_registry

app = typer.Typer(
    name="codebase-assistant",
    help="Intelligent Codebase Assistant — AI-powered code understanding",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(help="Question to ask the assistant"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (e.g., qwen2.5:latest, claude-sonnet-4-6)"
    ),
    repo_path: str = typer.Option(".", "--repo", "-r", help="Indexed repo path for code search"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show agent Thought/Action/Observation trace"
    ),
):
    """Ask the codebase assistant a question. Uses ReAct agent with code search tools."""
    try:
        provider = create_provider(model)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if verbose:
        console.print(f"[dim]Using: {provider.name()}[/dim]")

    registry = build_default_registry(Path(repo_path).resolve())
    agent = ReactAgent(provider=provider, registry=registry, verbose=verbose)

    try:
        with console.status("[bold green]Thinking..."):
            result = agent.run(question, system_prompt=AGENT_SYSTEM_PROMPT)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print()
    console.print(Markdown(result.answer))

    if verbose:
        console.print()
        console.print(
            f"[dim]Iterations: {result.iterations} | "
            f"Tools: {', '.join(result.tool_calls) or 'none'} | "
            f"Stopped: {result.stopped_reason}[/dim]"
        )


@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show debug info"
    ),
):
    """Start an interactive chat session with the assistant."""
    try:
        provider = create_provider(model)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"Model: {provider.name()}\nType 'quit' or 'exit' to end the session.",
            title="Codebase Assistant",
            border_style="green",
        )
    )

    # TODO: messages grow unbounded — Phase 4 (memory) will add
    # sliding window + summarization to stay within context limits.
    messages: list[Message] = []

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        messages.append(Message(role="user", content=user_input))

        try:
            with console.status("[bold green]Thinking..."):
                response = provider.chat(messages, system=SYSTEM_PROMPT)
        except RuntimeError as e:
            console.print(f"[red]Error:[/red] {e}")
            continue

        messages.append(Message(role="assistant", content=response.content))

        console.print()
        console.print(Markdown(response.content))
        console.print()

        if verbose:
            console.print(
                f"[dim]tokens: {response.usage['input_tokens']}in "
                f"/ {response.usage['output_tokens']}out[/dim]"
            )


@app.command()
def ingest(
    repo_path: str = typer.Argument(help="Path to the repository to index"),
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-file details"),
):
    """Index a codebase into the vector database for semantic search."""
    from codebase_assistant.rag.ingestion import CodebaseIngester

    path = Path(repo_path).resolve()
    if not path.is_dir():
        console.print(f"[red]Error:[/red] '{repo_path}' is not a directory")
        raise typer.Exit(code=1)

    ingester = CodebaseIngester(repo_path=path)

    try:
        with console.status("[bold green]Indexing codebase..."):
            stats = ingester.ingest(force=force)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if stats.new_files == 0 and stats.modified_files == 0 and stats.deleted_files == 0:
        console.print(
            Panel(
                f"[dim]All {stats.skipped_files} files unchanged — nothing to do.[/dim]",
                title="Index up to date",
                border_style="dim",
            )
        )
        return

    console.print(
        Panel(
            f"[green]✓[/green] {stats.new_files} new  "
            f"[yellow]~[/yellow] {stats.modified_files} modified  "
            f"[red]✗[/red] {stats.deleted_files} deleted  "
            f"[dim]— {stats.skipped_files} unchanged[/dim]\n"
            f"Chunks indexed: [bold]{stats.total_chunks}[/bold]\n"
            f"Duration: {stats.duration_seconds:.1f}s",
            title=f"Indexed [bold]{path.name}[/bold]",
            border_style="green",
        )
    )


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    repo_path: str = typer.Option(".", "--repo", "-r", help="Repository path"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    language: Optional[str] = typer.Option(None, "--lang", "-l", help="Filter by language (e.g. python)"),
    chunk_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type (function/class/method)"),
):
    """Search indexed code for relevant snippets."""
    from codebase_assistant.rag.retrieval import CodebaseRetriever

    path = Path(repo_path).resolve()
    retriever = CodebaseRetriever(repo_path=path)

    if not retriever.collection_exists():
        console.print(
            f"[red]Error:[/red] No index found for '{path}'. "
            "Run: [bold]codebase-assistant ingest <repo-path>[/bold]"
        )
        raise typer.Exit(code=1)

    try:
        with console.status("[bold green]Searching..."):
            if language or chunk_type:
                results = retriever.search_with_filter(
                    query, language=language, chunk_type=chunk_type, top_k=top_k
                )
            else:
                results = retriever.search(query, top_k=top_k)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, result in enumerate(results, 1):
        label = f"[bold]{result.file_path}[/bold]"
        if result.name:
            label += f"  [cyan]{result.chunk_type}:[/cyan] [bold cyan]{result.name}[/bold cyan]"
        label += f"  [dim]lines {result.start_line}–{result.end_line}[/dim]"

        console.print(
            Panel(
                Syntax(
                    result.content,
                    result.language or "text",
                    line_numbers=True,
                    start_line=result.start_line,
                    theme="monokai",
                ),
                title=label,
                subtitle=f"[dim]score {result.score:.3f}[/dim]",
                border_style="blue",
            )
        )
        if i < len(results):
            console.print()


if __name__ == "__main__":
    app()
