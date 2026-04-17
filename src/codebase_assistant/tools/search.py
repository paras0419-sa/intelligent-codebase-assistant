"""`search_code` tool — semantic search over the indexed repo.

The retriever needs a `repo_path` to locate the right ChromaDB collection, but
we don't want that in the tool's JSON schema (it would let the LLM try to
search arbitrary repos, and it's session-level config, not per-call input).

Instead, `build_search_code_tool(repo_path)` captures the path in a closure and
returns a `ToolDefinition` the registry can use directly. The LLM only sees
`query`, `top_k`, and `language`.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from codebase_assistant.rag.retrieval import CodebaseRetriever, SearchResult
from codebase_assistant.tools.registry import ToolDefinition


SEARCH_CODE_DESCRIPTION = (
    "Semantic search over the indexed codebase. Returns the top-k code chunks "
    "(functions, classes, blocks) most relevant to a natural-language query, "
    "with file paths and line ranges. Use this to locate code by concept or "
    "behaviour rather than exact keyword."
)


class SearchCodeInput(BaseModel):
    query: str = Field(description="Natural language query to search the codebase")
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return (1-10)",
    )
    language: str | None = Field(
        default=None,
        description="Optional language filter, e.g. 'python', 'java', 'typescript'",
    )


def _format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."

    blocks: list[str] = []
    for r in results:
        header = f"{r.location()} [{r.chunk_type}"
        if r.name:
            header += f": {r.name}"
        header += f"] (score={r.score:.3f})"
        blocks.append(f"{header}\n{r.content}")
    return "\n---\n".join(blocks)


def build_search_code_tool(repo_path: Path) -> ToolDefinition:
    """Create a `search_code` ToolDefinition bound to a specific indexed repo."""
    retriever = CodebaseRetriever(repo_path=repo_path)

    def handler(args: SearchCodeInput) -> str:
        if args.language:
            results = retriever.search_with_filter(
                query=args.query,
                language=args.language,
                top_k=args.top_k,
            )
        else:
            results = retriever.search(query=args.query, top_k=args.top_k)
        return _format_results(results)

    return ToolDefinition(
        name="search_code",
        description=SEARCH_CODE_DESCRIPTION,
        input_model=SearchCodeInput,
        handler=handler,
    )
