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
# Review fix (#3): a 200-line read at ~80 chars/line was ~16KB, 4x over the
# 4000-char budget documented in the Phase 3 plan. Enforce a char cap on top
# of the line cap so a single long-lined file can't swamp the agent context.
READ_FILE_MAX_CHARS = 4000
LIST_DIR_MAX_ENTRIES = 200

# Review fix (#2): noise directories that would otherwise dominate recursive
# listings (and waste tokens even on flat listings of the repo root). Kept
# small and static here; once rag/ingestion.py's gitignore logic is reusable
# as a helper, route both through it for consistency.
NOISE_DIR_NAMES = frozenset({
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
})


def _is_noise(entry: Path) -> bool:
    """True if the entry is a hidden dotfile or a known noise directory."""
    name = entry.name
    if name.startswith("."):
        return True
    return entry.is_dir() and name in NOISE_DIR_NAMES


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

    # Review fix (#3): enforce the char cap after formatting. Trim whole lines
    # from the tail so the output stays syntactically intact and the "showing
    # lines N-M" header remains accurate — partial-line truncation would be
    # misleading to the agent.
    if len(body) > READ_FILE_MAX_CHARS:
        trimmed_lines: list[str] = []
        size = 0
        for line in body.splitlines():
            if size + len(line) + 1 > READ_FILE_MAX_CHARS:
                break
            trimmed_lines.append(line)
            size += len(line) + 1
        end = start + len(trimmed_lines) - 1
        body = "\n".join(trimmed_lines)
        body += f"\n... (truncated at {READ_FILE_MAX_CHARS} chars)"

    header = f"{args.path} (showing lines {start}-{end} of {total})"
    return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def _tag(entry: Path) -> str:
    return "[dir] " if entry.is_dir() else "[file]"


def _rel_display(target: Path, repo_root: Path) -> str:
    """Render target as a repo-relative string; '.' when target == repo_root.

    Review fix (#5): `Path.relative_to(...) or '.'` never fell through because
    `PosixPath('.')` is truthy. Use an explicit stringify + equality check.
    """
    rel = str(target.relative_to(repo_root))
    return "." if rel == "." else rel


def _list_flat(target: Path, repo_root: Path) -> str:
    try:
        # Review fix (#2): filter hidden and noise dirs out of flat listings
        # too — otherwise `list_directory(".")` on a dev checkout dumps .git,
        # .venv, __pycache__, etc., alongside real project files.
        children = sorted(
            (c for c in target.iterdir() if not _is_noise(c)),
            key=lambda p: (not p.is_dir(), p.name),
        )
    except OSError as exc:
        return f"Error: could not list '{target}': {exc}"

    if not children:
        return f"(empty directory: {_rel_display(target, repo_root)})"

    lines = [f"{_tag(c)} {c.name}" for c in children]
    return "\n".join(lines)


def _list_recursive(target: Path, repo_root: Path) -> str:
    # Review fix (#1 + #4): walk the tree manually with an explicit stack so
    # we (a) stop as soon as the entry cap is hit — no more materialising the
    # whole tree via `sorted(rglob("*"))` before truncating — and (b) prune
    # noise directories at the directory boundary so we never descend into
    # .git / .venv / node_modules at all.
    #
    # target_depth is computed once (was previously recomputed per iteration).
    target_resolved = target.resolve()
    target_depth = len(target_resolved.relative_to(repo_root).parts)

    lines: list[str] = []
    truncated = False

    # Stack holds directories still to visit; each `iterdir()` result is
    # sorted locally so siblings appear in a stable order without requiring
    # a global sort of the entire tree.
    stack: list[Path] = [target]

    while stack and len(lines) < LIST_DIR_MAX_ENTRIES:
        current = stack.pop(0)
        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except OSError:
            continue

        # Push directories in reverse so we visit them in sorted order on
        # subsequent iterations (depth-first by directory, files emitted
        # as we go).
        dirs_to_visit: list[Path] = []

        for child in children:
            if _is_noise(child):
                continue

            # Symlink guard: if `.resolve()` jumps outside the repo root,
            # drop the entry entirely rather than listing it.
            try:
                rel = child.resolve().relative_to(repo_root)
            except ValueError:
                continue

            depth = len(rel.parts) - target_depth - 1
            indent = "  " * max(depth, 0)
            lines.append(f"{indent}{_tag(child)} {child.name}")

            if len(lines) >= LIST_DIR_MAX_ENTRIES:
                truncated = True
                break

            if child.is_dir():
                dirs_to_visit.append(child)

        # Prepend remaining dirs so traversal stays depth-first-ish and
        # siblings keep their order.
        stack = dirs_to_visit + stack

    if not lines:
        return f"(empty directory: {_rel_display(target, repo_root)})"

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
