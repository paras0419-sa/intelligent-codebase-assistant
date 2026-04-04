"""CLI entrypoint using Typer.

Typer builds on Click but uses Python type hints for argument/option
definitions — less boilerplate, better IDE support. The `rich` integration
gives us colored output and markdown rendering for free.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from codebase_assistant.models.base import Message
from codebase_assistant.models.factory import create_provider
from codebase_assistant.prompts import SYSTEM_PROMPT

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
        None, "--model", "-m", help="Model to use (e.g., claude-sonnet-4-6, gpt-4o)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show debug info (model, tokens)"
    ),
):
    """Ask the codebase assistant a question."""
    try:
        provider = create_provider(model)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if verbose:
        console.print(f"[dim]Using: {provider.name()}[/dim]")

    messages = [Message(role="user", content=question)]

    try:
        with console.status("[bold green]Thinking..."):
            response = provider.chat(messages, system=SYSTEM_PROMPT)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print()
    console.print(Markdown(response.content))

    if verbose:
        console.print()
        console.print(
            Panel(
                f"Model: {response.model}\n"
                f"Input tokens: {response.usage['input_tokens']}\n"
                f"Output tokens: {response.usage['output_tokens']}",
                title="Usage",
                border_style="dim",
            )
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


if __name__ == "__main__":
    app()
