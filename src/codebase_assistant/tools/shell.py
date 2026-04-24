import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from codebase_assistant.tools.registry import ToolDefinition

ALLOWED_COMMANDS = {"pytest", "python", "uv", "git", "grep", "find", "cat", "ls", "head", "tail"}
_MAX_OUTPUT = 4000


class RunCommandInput(BaseModel):
    command: str = Field(description=f"Command to run. Allowed: {sorted(ALLOWED_COMMANDS)}")
    args: list[str] = Field(default=[], description="Arguments for the command")
    working_dir: str | None = Field(default=None, description="Working directory (defaults to repo root)")


def make_run_command_tool(repo_path: Path) -> ToolDefinition:
    def handler(args: RunCommandInput) -> str:
        if args.command not in ALLOWED_COMMANDS:
            return f"Error: '{args.command}' is not allowed. Allowed commands: {sorted(ALLOWED_COMMANDS)}"

        cwd = repo_path
        if args.working_dir:
            candidate = Path(args.working_dir)
            if not candidate.is_absolute():
                candidate = repo_path / candidate
            candidate = candidate.resolve()
            if not str(candidate).startswith(str(repo_path)):
                return f"Error: working_dir '{args.working_dir}' is outside the repository."
            cwd = candidate

        try:
            result = subprocess.run(
                [args.command] + args.args,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(cwd),
            )
            output = (result.stdout + result.stderr)[:_MAX_OUTPUT]
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command '{args.command}' timed out after 30 seconds."
        except FileNotFoundError:
            return f"Error: command '{args.command}' not found. Is it installed?"

    return ToolDefinition(
        name="run_command",
        description=(
            f"Run a sandboxed shell command. Allowed commands: {sorted(ALLOWED_COMMANDS)}. "
            "Output is capped at 4000 characters. Timeout: 30 seconds."
        ),
        input_model=RunCommandInput,
        handler=handler,
    )
