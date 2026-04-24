import json
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from codebase_assistant.models.base import ModelProvider, ToolUseBlock
from codebase_assistant.tools.registry import ToolRegistry

_console = Console()


@dataclass
class AgentResult:
    answer: str
    iterations: int
    tool_calls: list[str]
    stopped_reason: str  # "end_turn" | "max_iterations"


class ReactAgent:
    """ReAct agent: Thought → Action (tool call) → Observation loop.

    Uses the model's native tool-calling API (OpenAI-compatible via Ollama).
    No prompt-engineering of tool schemas — the API handles structured dispatch.
    """

    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._max_iterations = max_iterations
        self._verbose = verbose

    def run(self, user_query: str, system_prompt: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": user_query}]
        tools = self._registry.to_openai_tools()
        tool_calls_made: list[str] = []

        for iteration in range(self._max_iterations):
            if self._verbose:
                _console.print(f"[dim][Iteration {iteration + 1}][/dim]")

            response = self._provider.chat_with_tools(
                messages=messages,
                system=system_prompt,
                tools=tools,
            )

            if response.stop_reason == "end_turn":
                answer = response.text()
                return AgentResult(answer, iteration + 1, tool_calls_made, "end_turn")

            if response.stop_reason == "tool_use":
                # Reconstruct the assistant message with tool_calls for history
                tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
                text_content = response.text()

                assistant_msg: dict = {"role": "assistant", "content": text_content or ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {"name": b.name, "arguments": json.dumps(b.input)},
                    }
                    for b in tool_use_blocks
                ]
                messages.append(assistant_msg)

                for block in tool_use_blocks:
                    tool_calls_made.append(block.name)

                    if self._verbose:
                        self._print_action(block.name, block.input)

                    result = self._registry.execute(block.name, block.input)

                    if self._verbose:
                        self._print_observation(result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.id,
                        "content": result,
                    })

        return AgentResult(
            "I was unable to fully answer within the allowed number of steps.",
            self._max_iterations,
            tool_calls_made,
            "max_iterations",
        )

    def _print_action(self, name: str, args: dict) -> None:
        args_str = json.dumps(args, indent=2)
        _console.print(
            Panel(
                Text(args_str, style="cyan"),
                title=f"[bold]Action:[/bold] {name}",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def _print_observation(self, result: str) -> None:
        preview = result[:500] + ("..." if len(result) > 500 else "")
        _console.print(
            Panel(
                Text(preview, style="dim"),
                title="[bold]Observation[/bold]",
                border_style="dim",
                padding=(0, 1),
            )
        )
