from pathlib import Path

from pydantic import BaseModel, Field

from codebase_assistant.rag.retrieval import CodebaseRetriever, SearchResult
from codebase_assistant.tools.registry import ToolDefinition


class SearchCodeInput(BaseModel):
    query: str = Field(description="Natural language query to search the codebase")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of results to return (1-10)")
    language: str | None = Field(default=None, description="Filter by language: python, java, typescript")


def _format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    parts = []
    for r in results:
        header = f"{r.file_path}:{r.start_line}-{r.end_line}"
        if r.name:
            header += f" ({r.chunk_type}: {r.name})"
        parts.append(f"{header}\n{r.content}")
    return "\n---\n".join(parts)


def make_search_tool(repo_path: Path) -> ToolDefinition:
    retriever = CodebaseRetriever(repo_path=repo_path)

    def handler(args: SearchCodeInput) -> str:
        if not retriever.collection_exists():
            return f"No index found for '{repo_path}'. Run: codebase-assistant ingest <repo-path>"
        if args.language:
            results = retriever.search_with_filter(args.query, language=args.language, top_k=args.top_k)
        else:
            results = retriever.search(args.query, top_k=args.top_k)
        return _format_results(results)

    return ToolDefinition(
        name="search_code",
        description=(
            "Search the indexed codebase using a natural language query. "
            "Returns relevant code snippets with file paths and line numbers. "
            "Use this before read_file to find what you're looking for."
        ),
        input_model=SearchCodeInput,
        handler=handler,
    )
