from pathlib import Path

from pydantic import BaseModel, Field

from codebase_assistant.tools.registry import ToolDefinition

_MAX_LINES = 200
_MAX_ENTRIES = 200


class ReadFileInput(BaseModel):
    path: str = Field(description="Absolute or repo-relative path to the file")
    start_line: int | None = Field(default=None, description="First line to read (1-indexed, inclusive)")
    end_line: int | None = Field(default=None, description="Last line to read (1-indexed, inclusive)")


class ListDirectoryInput(BaseModel):
    path: str = Field(description="Directory path to list")
    recursive: bool = Field(default=False, description="List directory contents recursively")


def make_read_file_tool(repo_path: Path) -> ToolDefinition:
    def handler(args: ReadFileInput) -> str:
        target = Path(args.path)
        if not target.is_absolute():
            target = repo_path / target
        target = target.resolve()

        if not str(target).startswith(str(repo_path)):
            return f"Error: path '{args.path}' is outside the repository."
        if not target.exists():
            return f"Error: file '{args.path}' does not exist."
        if not target.is_file():
            return f"Error: '{args.path}' is not a file."

        lines = target.read_text(errors="replace").splitlines()
        total = len(lines)

        start = (args.start_line - 1) if args.start_line else 0
        end = args.end_line if args.end_line else total
        start = max(0, start)
        end = min(total, end)

        # Cap to _MAX_LINES
        if end - start > _MAX_LINES:
            end = start + _MAX_LINES
            truncated = True
        else:
            truncated = False

        numbered = "\n".join(f"{i + 1:4d}  {line}" for i, line in enumerate(lines[start:end], start=start))
        note = f"\n[showing lines {start + 1}–{end} of {total} total]" if truncated else ""
        return numbered + note

    return ToolDefinition(
        name="read_file",
        description=(
            "Read a file from the repository. Optionally specify start_line and end_line "
            "to read a specific range. Returns content with line numbers."
        ),
        input_model=ReadFileInput,
        handler=handler,
    )


def make_list_directory_tool(repo_path: Path) -> ToolDefinition:
    def handler(args: ListDirectoryInput) -> str:
        target = Path(args.path)
        if not target.is_absolute():
            target = repo_path / target
        target = target.resolve()

        if not str(target).startswith(str(repo_path)):
            return f"Error: path '{args.path}' is outside the repository."
        if not target.exists():
            return f"Error: path '{args.path}' does not exist."
        if not target.is_dir():
            return f"Error: '{args.path}' is not a directory."

        if args.recursive:
            entries = sorted(target.rglob("*"))
            lines = []
            for e in entries[:_MAX_ENTRIES]:
                rel = e.relative_to(target)
                tag = "[dir]" if e.is_dir() else "[file]"
                lines.append(f"{tag} {rel}")
            result = "\n".join(lines)
            if len(entries) > _MAX_ENTRIES:
                result += f"\n[truncated — showing {_MAX_ENTRIES} of {len(entries)} entries]"
            return result
        else:
            entries = sorted(target.iterdir())
            lines = [
                f"{'[dir] ' if e.is_dir() else '[file]'} {e.name}"
                for e in entries
            ]
            return "\n".join(lines) if lines else "(empty directory)"

    return ToolDefinition(
        name="list_directory",
        description=(
            "List the contents of a directory. Use recursive=true to get a full tree. "
            "Output is capped at 200 entries."
        ),
        input_model=ListDirectoryInput,
        handler=handler,
    )
