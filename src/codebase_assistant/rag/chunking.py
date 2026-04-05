"""AST-aware code chunker using tree-sitter.

Why AST-aware instead of naive line/token splitting?
- Naive chunking breaks mid-function, mid-class — embeddings lose semantic meaning.
- AST chunking extracts whole functions, classes, methods as atomic units.
- Each chunk is a complete, self-contained piece of code the LLM can reason about.

How tree-sitter works:
- tree-sitter is a C-based incremental parser with language-specific grammars.
- It produces a concrete syntax tree (CST) — similar to an AST but includes
  punctuation, whitespace, and comments as unnamed nodes.
- We walk the top-level named nodes (function_definition, class_definition, etc.)
  and extract their source text as chunks.
- Decorators are part of `decorated_definition` nodes that wrap the actual
  function/class, so they stay attached automatically.
"""

from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from codebase_assistant.config import settings


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A single chunk of code extracted from a source file.

    This is our domain type — it isolates the rest of the codebase from
    tree-sitter internals, just like Message/ModelResponse isolate from
    LLM SDK types.
    """

    content: str
    file_path: str  # relative to repo root
    language: str
    chunk_type: str  # "function" | "class" | "method" | "module_preamble" | "block"
    name: str  # symbol name (e.g., "validate_email") or "" for preamble/block
    start_line: int  # 1-indexed
    end_line: int


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# Maps file extensions to tree-sitter language names.
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
}

# Top-level AST node types we extract as chunks, per language.
# These are the "semantic boundaries" — each becomes one chunk.
_CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
}

# Parser cache — one parser per language, created lazily.
_parsers: dict[str, tree_sitter.Parser] = {}


# ---------------------------------------------------------------------------
# Parser management
# ---------------------------------------------------------------------------

def _get_parser(language: str) -> tree_sitter.Parser:
    """Get or create a tree-sitter parser for the given language.

    Parsers are cached because creating one involves loading a C shared
    library — not expensive, but no reason to do it repeatedly.

    The tree-sitter 0.22+ API uses Language objects from individual
    language packages (tree-sitter-python, etc.) instead of the old
    .so file approach.
    """
    if language in _parsers:
        return _parsers[language]

    lang_obj = _load_language(language)
    parser = tree_sitter.Parser(lang_obj)
    _parsers[language] = parser
    return parser


def _load_language(language: str) -> tree_sitter.Language:
    """Load a tree-sitter Language from the corresponding package.

    Each language has its own PyPI package that exposes a `language()`
    function returning a raw pointer. We wrap it in tree_sitter.Language.
    """
    if language == "python":
        import tree_sitter_python
        return tree_sitter.Language(tree_sitter_python.language())

    # Future: add more languages here
    # if language == "java":
    #     import tree_sitter_java
    #     return tree_sitter.Language(tree_sitter_java.language())

    raise ValueError(
        f"No tree-sitter grammar installed for '{language}'. "
        f"Supported: {list(_CHUNK_NODE_TYPES.keys())}"
    )


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

def _extract_name(node: tree_sitter.Node) -> str:
    """Extract the symbol name from an AST node.

    For decorated_definition, we drill into the inner function/class.
    For function_definition/class_definition, we find the 'name' field.
    """
    # decorated_definition wraps the real function/class
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _extract_name(child)
        return ""

    # Look for the 'name' field (tree-sitter field, not just any child)
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode()

    return ""


def _classify_chunk_type(node: tree_sitter.Node) -> str:
    """Determine the chunk_type string from an AST node type."""
    # Unwrap decorated_definition to see what's inside
    actual = node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                actual = child
                break

    if actual.type == "class_definition":
        return "class"
    if actual.type == "function_definition":
        return "function"
    return "block"


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _extract_ast_chunks(
    source: bytes, file_path: str, language: str
) -> list[CodeChunk]:
    """Walk the tree-sitter AST and extract top-level semantic nodes.

    Strategy:
    1. Parse the source with tree-sitter.
    2. Iterate over top-level children of the root node.
    3. If a child is a chunk-worthy node type (function, class, decorated),
       extract it as a CodeChunk.
    4. For classes, also extract individual methods as separate chunks so
       you can search for a specific method, not just the entire class.
    5. Consecutive non-chunk nodes (imports, assignments, if-guards) are
       collected into a "module_preamble" chunk.
    """
    parser = _get_parser(language)
    tree = parser.parse(source)
    root = tree.root_node

    chunk_types = _CHUNK_NODE_TYPES.get(language, set())
    chunks: list[CodeChunk] = []
    preamble_lines: list[tuple[int, int]] = []  # (start, end) ranges for non-chunk nodes

    source_lines = source.decode(errors="replace").splitlines(keepends=True)

    for child in root.children:
        if not child.is_named:
            continue

        if child.type in chunk_types:
            # Flush any accumulated preamble before this chunk
            if preamble_lines:
                chunks.append(
                    _make_preamble_chunk(source_lines, preamble_lines, file_path, language)
                )
                preamble_lines = []

            # Extract the main chunk (entire function or class)
            chunks.append(_node_to_chunk(child, source_lines, file_path, language))

            # For classes, also extract individual methods
            if _classify_chunk_type(child) == "class":
                chunks.extend(
                    _extract_methods(child, source_lines, file_path, language)
                )
        else:
            # Non-chunk node — accumulate for preamble
            preamble_lines.append((child.start_point[0], child.end_point[0]))

    # Flush remaining preamble at end of file
    if preamble_lines:
        chunks.append(
            _make_preamble_chunk(source_lines, preamble_lines, file_path, language)
        )

    return chunks


def _node_to_chunk(
    node: tree_sitter.Node,
    source_lines: list[str],
    file_path: str,
    language: str,
) -> CodeChunk:
    """Convert a tree-sitter node into a CodeChunk."""
    start = node.start_point[0]  # 0-indexed line
    end = node.end_point[0]
    content = "".join(source_lines[start : end + 1])

    return CodeChunk(
        content=content,
        file_path=file_path,
        language=language,
        chunk_type=_classify_chunk_type(node),
        name=_extract_name(node),
        start_line=start + 1,  # convert to 1-indexed
        end_line=end + 1,
    )


def _extract_methods(
    class_node: tree_sitter.Node,
    source_lines: list[str],
    file_path: str,
    language: str,
) -> list[CodeChunk]:
    """Extract individual methods from a class as separate chunks.

    This enables searching for a specific method rather than only the
    whole class. Each method chunk has chunk_type="method".
    """
    methods: list[CodeChunk] = []

    # For decorated_definition, drill into the class body
    body_node = class_node
    if class_node.type == "decorated_definition":
        for child in class_node.children:
            if child.type == "class_definition":
                body_node = child
                break

    # Find the block/body child of the class
    block = body_node.child_by_field_name("body")
    if not block:
        return methods

    for child in block.children:
        if not child.is_named:
            continue
        if child.type in ("function_definition", "decorated_definition"):
            chunk = _node_to_chunk(child, source_lines, file_path, language)
            chunk.chunk_type = "method"
            methods.append(chunk)

    return methods


def _make_preamble_chunk(
    source_lines: list[str],
    line_ranges: list[tuple[int, int]],
    file_path: str,
    language: str,
) -> CodeChunk:
    """Combine non-chunk line ranges into a module_preamble chunk.

    This captures imports, module-level constants, if __name__ guards, etc.
    """
    start = line_ranges[0][0]
    end = line_ranges[-1][1]
    content = "".join(source_lines[start : end + 1])

    return CodeChunk(
        content=content,
        file_path=file_path,
        language=language,
        chunk_type="module_preamble",
        name="",
        start_line=start + 1,
        end_line=end + 1,
    )


# ---------------------------------------------------------------------------
# Large/small chunk handling
# ---------------------------------------------------------------------------

def _split_large_chunk(chunk: CodeChunk, max_lines: int) -> list[CodeChunk]:
    """Split a chunk that exceeds max_lines at blank-line boundaries.

    Why blank lines? They're natural paragraph breaks in code — between
    logical blocks inside a function, between method groups inside a class.
    Splitting here preserves readability better than splitting at arbitrary lines.
    """
    lines = chunk.content.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return [chunk]

    # Find blank-line positions for natural split points
    split_points = [i for i, line in enumerate(lines) if line.strip() == ""]

    parts: list[CodeChunk] = []
    current_start = 0
    part_index = 0

    while current_start < len(lines):
        target = current_start + max_lines
        if target >= len(lines):
            # Last segment — take everything remaining
            segment_end = len(lines)
        else:
            # Find the nearest blank line at or before the target
            best_split = target
            for sp in split_points:
                if current_start < sp <= target:
                    best_split = sp
            segment_end = best_split

        segment = lines[current_start:segment_end]
        if not segment:
            break

        abs_start = chunk.start_line + current_start
        abs_end = abs_start + len(segment) - 1

        parts.append(CodeChunk(
            content="".join(segment),
            file_path=chunk.file_path,
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            name=f"{chunk.name}[part {part_index}]" if chunk.name else "",
            start_line=abs_start,
            end_line=abs_end,
        ))

        current_start = segment_end
        part_index += 1

    return parts


def _group_small_chunks(chunks: list[CodeChunk], min_lines: int) -> list[CodeChunk]:
    """Merge adjacent small chunks (< min_lines) into larger ones.

    Why? Thousands of 1-3 line chunks waste embedding storage and reduce
    retrieval quality — the embedding of "x = 42" alone has no useful
    semantic signal.

    Only groups chunks of the same chunk_type that are adjacent in the file.
    Never groups across different types (don't merge a function into preamble).
    """
    if not chunks:
        return chunks

    result: list[CodeChunk] = []
    buffer: CodeChunk | None = None

    for chunk in chunks:
        line_count = chunk.end_line - chunk.start_line + 1

        if line_count >= min_lines:
            # Big enough on its own — flush buffer first
            if buffer:
                result.append(buffer)
                buffer = None
            result.append(chunk)
            continue

        # Small chunk — try to merge into buffer
        if buffer and buffer.chunk_type == chunk.chunk_type:
            buffer = CodeChunk(
                content=buffer.content + chunk.content,
                file_path=buffer.file_path,
                language=buffer.language,
                chunk_type=buffer.chunk_type,
                name=buffer.name if buffer.name else chunk.name,
                start_line=buffer.start_line,
                end_line=chunk.end_line,
            )
            # If merged buffer is now big enough, flush it
            if buffer.end_line - buffer.start_line + 1 >= min_lines:
                result.append(buffer)
                buffer = None
        else:
            # Different type — flush old buffer, start new one
            if buffer:
                result.append(buffer)
            buffer = chunk

    if buffer:
        result.append(buffer)

    return result


# ---------------------------------------------------------------------------
# Fallback chunker (non-AST)
# ---------------------------------------------------------------------------

def _chunk_by_blank_lines(
    source: str, file_path: str, language: str
) -> list[CodeChunk]:
    """Simple fallback chunker for unsupported languages.

    Splits at blank lines, grouping consecutive non-blank lines into blocks.
    Less precise than AST chunking but works for any text file.
    """
    lines = source.splitlines(keepends=True)
    chunks: list[CodeChunk] = []
    block_start: int | None = None
    block_lines: list[str] = []

    for i, line in enumerate(lines):
        if line.strip() == "":
            if block_lines:
                chunks.append(CodeChunk(
                    content="".join(block_lines),
                    file_path=file_path,
                    language=language,
                    chunk_type="block",
                    name="",
                    start_line=block_start + 1,  # type: ignore[operator]
                    end_line=block_start + len(block_lines),  # type: ignore[operator]
                ))
                block_lines = []
                block_start = None
        else:
            if block_start is None:
                block_start = i
            block_lines.append(line)

    # Flush final block
    if block_lines:
        chunks.append(CodeChunk(
            content="".join(block_lines),
            file_path=file_path,
            language=language,
            chunk_type="block",
            name="",
            start_line=block_start + 1,  # type: ignore[operator]
            end_line=block_start + len(block_lines),  # type: ignore[operator]
        ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_file(source: str, file_path: str, language: str) -> list[CodeChunk]:
    """Chunk a source file into semantically meaningful code chunks.

    This is the main entry point. It:
    1. Tries AST-aware chunking if a tree-sitter grammar is available.
    2. Falls back to blank-line chunking for unsupported languages.
    3. Splits oversized chunks and groups undersized ones.

    Args:
        source: The file's source code as a string.
        file_path: Path relative to repo root (stored in chunk metadata).
        language: Language name (e.g., "python"). Use detect_language() if unknown.
    """
    max_lines = settings.max_chunk_lines
    min_lines = settings.min_chunk_lines

    if language in _CHUNK_NODE_TYPES:
        try:
            chunks = _extract_ast_chunks(source.encode(), file_path, language)
        except Exception:
            # If parsing fails (e.g., syntax error), fall back gracefully
            chunks = _chunk_by_blank_lines(source, file_path, language)
    else:
        chunks = _chunk_by_blank_lines(source, file_path, language)

    # Post-process: split large chunks, group small ones
    processed: list[CodeChunk] = []
    for chunk in chunks:
        processed.extend(_split_large_chunk(chunk, max_lines))

    processed = _group_small_chunks(processed, min_lines)

    return processed


def chunk_file_from_path(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    """Convenience: read a file, detect language, and chunk it.

    Args:
        file_path: Absolute path to the source file.
        repo_root: Absolute path to the repository root.

    Returns:
        List of CodeChunks with file_path relative to repo_root.
    """
    relative_path = str(file_path.relative_to(repo_root))
    language = detect_language(file_path)
    source = file_path.read_text(errors="replace")

    if not source.strip():
        return []

    return chunk_file(source, relative_path, language)


def detect_language(file_path: Path) -> str:
    """Detect language from file extension."""
    return LANGUAGE_MAP.get(file_path.suffix.lower(), "unknown")
