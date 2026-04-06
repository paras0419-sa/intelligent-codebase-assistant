"""Codebase ingestion pipeline: walk repo → chunk → embed → store in ChromaDB.

Pipeline overview:
    1. Walk the repo with os.walk, filter using .gitignore (via pathspec)
       and a hardcoded exclude list (node_modules, __pycache__, etc.)
    2. Compute SHA-256 hashes for all eligible files
    3. Diff against the stored manifest to find added / modified / deleted files
    4. Delete ChromaDB documents for modified + deleted files
    5. Chunk and embed only the changed files (batch of 64 chunks at a time)
    6. Save the updated manifest to disk

Why file-hash manifests instead of git diff?
- Git only tracks committed changes; uncommitted edits would be missed.
- Works on non-git directories (build outputs, scratch folders, etc.).
- No gitpython dependency needed — just hashlib + json.

Why ChromaDB PersistentClient?
- Data lives on disk, not in-process memory — survives restarts.
- Zero config for local dev; swap to a network Chroma server later without
  changing any application code (just change the client constructor).

Document ID scheme: "{relative_file_path}::{start_line}-{end_line}"
- Deterministic — re-ingesting the same file produces identical IDs.
- Allows targeted deletes by file path prefix.
- Human-readable when browsing the ChromaDB storage directly.
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import pathspec

from codebase_assistant.config import settings
from codebase_assistant.rag.chunking import (
    LANGUAGE_MAP,
    CodeChunk,
    chunk_file_from_path,
)
from codebase_assistant.rag.embeddings import EmbeddingProvider, create_embedding_provider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directories always excluded regardless of .gitignore
_ALWAYS_EXCLUDE: set[str] = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "target",          # build outputs
    ".idea", ".vscode",                  # IDE dirs
}

# File extensions we want to index (code + light docs)
_INDEXABLE_EXTENSIONS: set[str] = set(LANGUAGE_MAP.keys()) | {
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
}

# How many chunks to embed + store per ChromaDB batch call
_EMBED_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class IngestionStats:
    """Statistics from a single ingestion run."""
    total_files: int = 0
    new_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    skipped_files: int = 0      # unchanged
    total_chunks: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Files: {self.total_files} total | "
            f"{self.new_files} new, {self.modified_files} modified, "
            f"{self.deleted_files} deleted, {self.skipped_files} unchanged\n"
            f"Chunks indexed: {self.total_chunks}\n"
            f"Duration: {self.duration_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------

def _collection_name(repo_path: Path) -> str:
    """Derive a stable ChromaDB collection name from the repo path.

    ChromaDB collection names must match [a-zA-Z0-9_-]{3,63}.
    We sanitize the absolute path into a safe string and truncate if needed.

    Example: /Users/paras/projects/my-app → repo__Users_paras_projects_my_app
    """
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", str(repo_path))
    name = f"repo__{sanitized}"
    # ChromaDB max is 63 chars; keep the tail (most specific part)
    return name[-63:] if len(name) > 63 else name


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_path(collection_name: str) -> Path:
    base = Path(settings.index_path).expanduser() / collection_name
    base.mkdir(parents=True, exist_ok=True)
    return base / "manifest.json"


def _load_manifest(path: Path) -> dict[str, str]:
    """Load {relative_file_path: sha256_hash} from disk."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def _sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Repo walker
# ---------------------------------------------------------------------------

def _load_gitignore(repo_root: Path) -> pathspec.PathSpec:
    """Parse the root .gitignore into a PathSpec matcher.

    pathspec supports the same glob patterns as git:
    - Wildcards: *.pyc, build/
    - Negation: !important.py
    - Directory markers: logs/ only matches directories

    We only read the root .gitignore for now (nested ones are a future
    enhancement — they're rarely needed for standard repos).
    """
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return pathspec.PathSpec.from_lines("gitwildmatch", [])
    return pathspec.PathSpec.from_lines(
        "gitwildmatch", gitignore.read_text().splitlines()
    )


def _walk_repo(repo_root: Path) -> list[Path]:
    """Return all indexable files under repo_root.

    Filters:
    1. Skip directories in _ALWAYS_EXCLUDE.
    2. Skip paths matched by .gitignore.
    3. Only include files with extensions in _INDEXABLE_EXTENSIONS.
    """
    ignore_spec = _load_gitignore(repo_root)
    result: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        current_dir = Path(dirpath)

        # Prune excluded directories in-place (modifies os.walk's traversal)
        dirnames[:] = [
            d for d in dirnames
            if d not in _ALWAYS_EXCLUDE
            and not ignore_spec.match_file(
                str((current_dir / d).relative_to(repo_root)) + "/"
            )
        ]

        for filename in filenames:
            file_path = current_dir / filename
            relative = file_path.relative_to(repo_root)

            if ignore_spec.match_file(str(relative)):
                continue
            if file_path.suffix.lower() not in _INDEXABLE_EXTENSIONS:
                continue

            result.append(file_path)

    return sorted(result)


# ---------------------------------------------------------------------------
# Main ingester
# ---------------------------------------------------------------------------

class CodebaseIngester:
    """Orchestrates the full ingestion pipeline for a single repository.

    Usage:
        ingester = CodebaseIngester(repo_path=Path("/path/to/repo"))
        stats = ingester.ingest()
        print(stats.summary())
    """

    def __init__(
        self,
        repo_path: Path,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._repo_path = repo_path.resolve()
        self._embedder = embedding_provider or create_embedding_provider()
        self._col_name = _collection_name(self._repo_path)
        self._manifest_path = _manifest_path(self._col_name)

        # PersistentClient stores data on disk — survives restarts
        self._chroma = chromadb.PersistentClient(
            path=str(Path(settings.chromadb_path).expanduser())
        )

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create a new one.

        We use cosine distance because:
        - Embedding models output L2-normalized vectors for text.
        - Cosine similarity is invariant to vector magnitude, making it
          more stable than L2 distance when comparing texts of different lengths.
        - nomic-embed-text and most other models are optimized for cosine.
        """
        return self._chroma.get_or_create_collection(
            name=self._col_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, force: bool = False) -> IngestionStats:
        """Run the full ingestion pipeline.

        Args:
            force: If True, re-index all files even if unchanged.

        Returns:
            IngestionStats with counts of processed files and chunks.
        """
        start_time = time.time()
        stats = IngestionStats()

        # 1. Walk repo
        all_files = _walk_repo(self._repo_path)
        stats.total_files = len(all_files)

        # 2. Compute current hashes (relative path → hash)
        current: dict[str, str] = {}
        for fp in all_files:
            rel = str(fp.relative_to(self._repo_path))
            current[rel] = _sha256(fp)

        # 3. Load manifest and diff
        manifest = {} if force else _load_manifest(self._manifest_path)
        added, modified, deleted = self._diff(current, manifest)

        stats.new_files = len(added)
        stats.modified_files = len(modified)
        stats.deleted_files = len(deleted)
        stats.skipped_files = stats.total_files - len(added) - len(modified)

        if not added and not modified and not deleted:
            stats.duration_seconds = time.time() - start_time
            return stats  # nothing to do

        collection = self._get_or_create_collection()

        # 4. Delete documents for modified + deleted files
        to_remove = modified + deleted
        if to_remove:
            # ChromaDB where filter: match any doc whose file_path is in the list
            collection.delete(where={"file_path": {"$in": to_remove}})

        # 5. Chunk + embed + store for added + modified files
        to_process = added + modified
        if to_process:
            stats.total_chunks = self._process_files(to_process, collection)

        # 6. Save updated manifest
        for rel in deleted:
            manifest.pop(rel, None)
        manifest.update(current)
        _save_manifest(self._manifest_path, manifest)

        stats.duration_seconds = time.time() - start_time
        return stats

    def _diff(
        self,
        current: dict[str, str],
        manifest: dict[str, str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Compare current file hashes against the stored manifest.

        Returns:
            (added, modified, deleted) — lists of relative file paths.
        """
        current_set = set(current)
        manifest_set = set(manifest)

        added = sorted(current_set - manifest_set)
        deleted = sorted(manifest_set - current_set)
        modified = sorted(
            p for p in current_set & manifest_set
            if current[p] != manifest[p]
        )
        return added, modified, deleted

    def _process_files(
        self,
        relative_paths: list[str],
        collection: chromadb.Collection,
    ) -> int:
        """Chunk, embed, and store a list of files. Returns total chunk count."""
        total_chunks = 0
        buffer: list[CodeChunk] = []

        def flush(buf: list[CodeChunk]) -> None:
            """Embed a batch of chunks and upsert into ChromaDB."""
            if not buf:
                return

            texts = [c.content for c in buf]
            vectors = self._embedder.embed(texts)

            ids = [_chunk_id(c) for c in buf]
            metadatas = [_chunk_metadata(c) for c in buf]

            # upsert = insert or update — safe to call repeatedly
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )

        for rel in relative_paths:
            file_path = self._repo_path / rel
            try:
                chunks = chunk_file_from_path(file_path, self._repo_path)
            except Exception:
                # Skip files that fail to parse (binary, encoding issues, etc.)
                continue

            for chunk in chunks:
                buffer.append(chunk)
                if len(buffer) >= _EMBED_BATCH_SIZE:
                    flush(buffer)
                    total_chunks += len(buffer)
                    buffer = []

        # Flush remaining chunks
        flush(buffer)
        total_chunks += len(buffer)

        return total_chunks

    def collection_name(self) -> str:
        """Return the ChromaDB collection name for this repo."""
        return self._col_name


# ---------------------------------------------------------------------------
# Chunk ID and metadata helpers
# ---------------------------------------------------------------------------

def _chunk_id(chunk: CodeChunk) -> str:
    """Deterministic document ID for a chunk.

    Format: "relative/path/to/file.py::10-45"
    This lets us:
    - Delete all chunks for a file using a prefix scan
    - Upsert the same chunk safely (idempotent)
    """
    return f"{chunk.file_path}::{chunk.start_line}-{chunk.end_line}"


def _chunk_metadata(chunk: CodeChunk) -> dict:
    """Serialize a CodeChunk's metadata for ChromaDB storage.

    ChromaDB metadata values must be str, int, float, or bool.
    We store everything needed to reconstruct a SearchResult without
    re-reading the source file.
    """
    return {
        "file_path": chunk.file_path,
        "language": chunk.language,
        "chunk_type": chunk.chunk_type,
        "name": chunk.name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }
