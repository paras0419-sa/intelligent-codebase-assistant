"""Tool registry for the ReAct agent.

Each tool is a `ToolDefinition` pairing:
    - a name + human description (what the LLM sees)
    - a Pydantic input model (auto-generates JSON Schema for the Claude tools API
      and validates raw LLM input before the handler runs)
    - a handler callable (input_model -> str)

The registry exposes three operations the agent loop needs:

    to_claude_tools()  → list[dict] in the shape Anthropic's `tools=` parameter expects
    execute(name, raw) → validates `raw` against the input model and calls the handler
    get(name)          → lookup for introspection / verbose printing

Validation failures and unknown tools are returned as string error messages so the
agent can observe them and adapt — never raised to the caller.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_claude_tools(self) -> list[dict]:
        """Serialize all tools in the shape Anthropic's messages API expects."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_model.model_json_schema(),
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, raw_input: dict) -> str:
        """Validate `raw_input` against the tool's schema and invoke its handler.

        Errors are stringified so the agent loop can feed them back as an
        observation instead of crashing the run.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available tools: {', '.join(self.names())}"

        try:
            validated = tool.input_model.model_validate(raw_input)
        except ValidationError as exc:
            return f"Error: invalid input for '{name}': {exc}"

        try:
            return tool.handler(validated)
        except Exception as exc:  # noqa: BLE001 — surface any handler failure to the agent
            return f"Error: tool '{name}' failed: {exc}"
