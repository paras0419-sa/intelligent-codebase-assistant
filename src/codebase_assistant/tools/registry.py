from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel


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
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def to_openai_tools(self) -> list[dict]:
        """Return tool definitions in OpenAI function-calling format (also used by Ollama)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_model.model_json_schema(),
                },
            }
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def execute(self, name: str, raw_input: dict) -> str:
        """Validate raw_input against the tool's schema and invoke its handler."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            parsed = tool.input_model.model_validate(raw_input)
            return tool.handler(parsed)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"


def build_default_registry(repo_path: Path) -> ToolRegistry:
    from codebase_assistant.tools.filesystem import make_list_directory_tool, make_read_file_tool
    from codebase_assistant.tools.search import make_search_tool
    from codebase_assistant.tools.shell import make_run_command_tool

    registry = ToolRegistry()
    registry.register(make_search_tool(repo_path))
    registry.register(make_read_file_tool(repo_path))
    registry.register(make_list_directory_tool(repo_path))
    registry.register(make_run_command_tool(repo_path))
    return registry
