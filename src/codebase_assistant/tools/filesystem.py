"""Filesystem tools: `read_file` and `list_directory`.

Both tools are sandboxed to a single `repo_path` captured at build time — the
LLM cannot escape the repo via relative paths (`../../etc/passwd`) or absolute
paths pointing elsewhere. Path traversal is caught by resolving the target and
checking it is still inside the repo root.

Outputs are size-capped so a malicious or overly large file/directory can't
blow up the agent's context window. File reads are annotated with line numbers
so the model can follow up with precise `start_line`/`end_line` ranges.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from codebase_assistant.tools.registry import ToolDefinition


# Output caps — tuned to the 4000-char per-tool budget from the Phase 3 plan.
READ_FILE_DEFAULT_MAX_LINES = 200
LIST_DIR_MAX_ENTRIES = 200


READ_FILE_DESCRIPTION = (
    "Read the contents of a file inside the indexed repo, with line numbers. "
    "Supports partial reads via start_line / end_line (1-indexed, inclusive). "
    "Large files are truncated; the output notes the shown range."
)

LIST_DIRECTORY_DESCRIPTION = (
    "List the contents of a directory inside the indexed repo. By default "
    "lists immediate children with type tags ([dir] / [file]). Set recursive=true "
    "to walk the tree (capped at 200 entries)."
)


class ReadFileInput(BaseModel):
    path: str = Field(description="Absolute or repo-relative path to the file")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="First line to read, 1-indexed",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Last line to read, inclusive",
    )


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Directory path, relative to repo root or absolute")
    recursive: bool = Field(default=False, description="List recursively as a tree")


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _resolve_within(repo_root: Path, raw: str) -> Path:
    """Resolve `raw` against `repo_root` and ensure the result stays inside it.

    Raises ValueError if the resolved path escapes the repo.
    """
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{raw}' resolves outside the repo root ({repo_root})"
        ) from exc

    return resolved


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def _format_file_lines(lines: list[str], first_line_number: int) -> str:
    width = len(str(first_line_number + len(lines) - 1))
    return "\n".join(
        f"{(first_line_number + i):>{width}} | {line.rstrip()}"
        for i, line in enumerate(lines)
    )


def _read_file(repo_root: Path, args: ReadFileInput) -> str:
    target = _resolve_within(repo_root, args.path)

    if not target.exists():
        return f"Error: file not found: {args.path}"
    if not target.is_file():
        return f"Error: not a regular file: {args.path}"

    try:
        all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Error: could not read '{args.path}': {exc}"

    total = len(all_lines)
    if total == 0:
        return f"(empty file: {args.path})"

    start = args.start_line or 1
    if start > total:
        return f"Error: start_line {start} is past end of file ({total} lines)"

    if args.end_line is not None:
        end = min(args.end_line, total)
    else:
        end = min(start + READ_FILE_DEFAULT_MAX_LINES - 1, total)

    if end < start:
        return f"Error: end_line {args.end_line} is before start_line {start}"

    shown = all_lines[start - 1 : end]
    body = _format_file_lines(shown, start)

    header = f"{args.path} (showing lines {start}-{end} of {total})"
    return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def _tag(entry: Path) -> str:
    return "[dir] " if entry.is_dir() else "[file]"


def _list_flat(target: Path, repo_root: Path) -> str:
    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError as exc:
        return f"Error: could not list '{target}': {exc}"

    if not children:
        return f"(empty directory: {target.relative_to(repo_root) or '.'})"

    lines = [f"{_tag(c)} {c.name}" for c in children]
    return "\n".join(lines)


def _list_recursive(target: Path, repo_root: Path) -> str:
    lines: list[str] = []
    count = 0
    truncated = False

    for path in sorted(target.rglob("*")):
        # Skip anything that slipped outside via symlinks.
        try:
            rel = path.resolve().relative_to(repo_root)
        except ValueError:
            continue

        depth = len(rel.parts) - len(target.resolve().relative_to(repo_root).parts) - 1
        indent = "  " * max(depth, 0)
        lines.append(f"{indent}{_tag(path)} {path.name}")

        count += 1
        if count >= LIST_DIR_MAX_ENTRIES:
            truncated = True
            break

    if not lines:
        return f"(empty directory: {target.relative_to(repo_root) or '.'})"

    output = "\n".join(lines)
    if truncated:
        output += f"\n... (truncated at {LIST_DIR_MAX_ENTRIES} entries)"
    return output


def _list_directory(repo_root: Path, args: ListDirectoryInput) -> str:
    target = _resolve_within(repo_root, args.path)

    if not target.exists():
        return f"Error: directory not found: {args.path}"
    if not target.is_dir():
        return f"Error: not a directory: {args.path}"

    if args.recursive:
        return _list_recursive(target, repo_root)
    return _list_flat(target, repo_root)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_read_file_tool(repo_path: Path) -> ToolDefinition:
    repo_root = repo_path.resolve()

    def handler(args: ReadFileInput) -> str:
        try:
            return _read_file(repo_root, args)
        except ValueError as exc:
            return f"Error: {exc}"

    return ToolDefinition(
        name="read_file",
        description=READ_FILE_DESCRIPTION,
        input_model=ReadFileInput,
        handler=handler,
    )


def build_list_directory_tool(repo_path: Path) -> ToolDefinition:
    repo_root = repo_path.resolve()

    def handler(args: ListDirectoryInput) -> str:
        try:
            return _list_directory(repo_root, args)
        except ValueError as exc:
            return f"Error: {exc}"

    return ToolDefinition(
        name="list_directory",
        description=LIST_DIRECTORY_DESCRIPTION,
        input_model=ListDirectoryInput,
        handler=handler,
    )
