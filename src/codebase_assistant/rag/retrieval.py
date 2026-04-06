"""Semantic search over an indexed codebase using ChromaDB.

How retrieval works:
1. The user's query string is embedded using the same model used during ingestion.
   This is critical — query and document embeddings must be in the same vector space.
2. ChromaDB's HNSW index performs approximate nearest neighbour (ANN) search,
   returning the top-k chunks whose embeddings are closest to the query embedding.
3. ChromaDB returns cosine distances (0 = identical, 2 = opposite). We convert
   these to similarity scores (1 = identical, 0 = unrelated) for readability.
4. Results are returned as SearchResult objects — our domain type, isolating
   callers from ChromaDB internals.

Optional metadata filters allow narrowing results by language or chunk_type
before the vector search, which ChromaDB handles efficiently via its WHERE clause.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from codebase_assistant.config import settings
from codebase_assistant.rag.embeddings import EmbeddingProvider, create_embedding_provider
from codebase_assistant.rag.ingestion import _collection_name


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single result from a semantic code search.

    Mirrors CodeChunk but adds a similarity score and is the public-facing
    type — callers (CLI, agent tools) import SearchResult, not CodeChunk.
    """

    content: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    score: float  # cosine similarity, 0.0–1.0 (higher = more relevant)

    def location(self) -> str:
        """Human-readable location string, e.g. 'src/models/base.py:10-45'."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CodebaseRetriever:
    """Semantic search over a ChromaDB-indexed codebase.

    Usage:
        retriever = CodebaseRetriever(repo_path=Path("/path/to/repo"))
        results = retriever.search("how does authentication work?")
        for r in results:
            print(r.location(), r.score)
    """

    def __init__(
        self,
        repo_path: Path,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._repo_path = repo_path.resolve()
        self._embedder = embedding_provider or create_embedding_provider()
        self._col_name = _collection_name(self._repo_path)
        self._chroma = chromadb.PersistentClient(
            path=str(Path(settings.chromadb_path).expanduser())
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search for code chunks relevant to the query.

        Args:
            query: Natural language or code query string.
            top_k: Number of results to return. Defaults to settings.search_top_k.

        Returns:
            List of SearchResult ordered by descending similarity score.
        """
        return self._query(query, top_k=top_k)

    def search_with_filter(
        self,
        query: str,
        language: str | None = None,
        chunk_type: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search with optional metadata filters.

        Filters are applied before the vector search (ChromaDB WHERE clause),
        so they narrow the candidate set rather than post-filtering results.

        Args:
            query: Natural language or code query string.
            language: Filter to a specific language, e.g. "python".
            chunk_type: Filter to a chunk type, e.g. "function", "class", "method".
            top_k: Number of results to return.
        """
        where: dict = {}
        conditions = []

        if language:
            conditions.append({"language": language})
        if chunk_type:
            conditions.append({"chunk_type": chunk_type})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            # ChromaDB uses MongoDB-style $and for multiple conditions
            where = {"$and": conditions}

        return self._query(query, top_k=top_k, where=where or None)

    def _query(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Internal query method — embeds query and calls ChromaDB."""
        k = top_k or settings.search_top_k

        try:
            collection = self._chroma.get_collection(self._col_name)
        except Exception:
            raise RuntimeError(
                f"No index found for '{self._repo_path}'. "
                "Run: codebase-assistant ingest <repo-path>"
            )

        query_vector = self._embedder.embed([query])[0]

        query_kwargs: dict = {
            "query_embeddings": [query_vector],
            "n_results": min(k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = collection.query(**query_kwargs)

        return _parse_results(raw)

    def collection_exists(self) -> bool:
        """Check whether this repo has been indexed."""
        try:
            self._chroma.get_collection(self._col_name)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_results(raw: dict) -> list[SearchResult]:
    """Convert ChromaDB query output into SearchResult objects.

    ChromaDB returns parallel lists:
        raw["ids"]       → [["id1", "id2", ...]]   (nested — one list per query)
        raw["documents"] → [["content1", ...]]
        raw["metadatas"] → [[{...}, ...]]
        raw["distances"] → [[0.12, 0.34, ...]]      (cosine distance, lower = better)

    We flatten the outer list (we always send one query at a time) and
    convert cosine distance → similarity score:
        similarity = 1 - (distance / 2)
    This maps [0, 2] → [1, 0] where 1.0 means identical vectors.
    """
    results: list[SearchResult] = []

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Cosine distance in ChromaDB ranges [0, 2]; convert to similarity [1, 0]
        score = round(1.0 - dist / 2.0, 4)
        results.append(SearchResult(
            content=doc,
            file_path=meta.get("file_path", ""),
            language=meta.get("language", ""),
            chunk_type=meta.get("chunk_type", ""),
            name=meta.get("name", ""),
            start_line=int(meta.get("start_line", 0)),
            end_line=int(meta.get("end_line", 0)),
            score=score,
        ))

    return results
