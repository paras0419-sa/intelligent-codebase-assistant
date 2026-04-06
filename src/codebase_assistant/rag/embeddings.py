"""Embedding provider abstraction for the RAG pipeline.

Why a separate EmbeddingProvider ABC instead of reusing ModelProvider?
- Chat and embedding have fundamentally different interfaces:
    chat(messages) -> text response
    embed(texts)   -> list of float vectors
- Mixing them into one ABC would violate the Single Responsibility Principle
  and force every chat provider to stub out embed() or vice versa.
- Keeping them separate mirrors how the APIs themselves are separate:
  Anthropic has no embedding API at all; OpenAI has a dedicated /embeddings
  endpoint; Ollama exposes both under the same base URL.

Provider implementations follow the same pattern as the chat providers:
- ABC for the contract
- Concrete classes for each backend
- Factory function reads config and returns the right implementation
- Lazy imports so SDK modules only load when that provider is used

Batching strategy:
- embed() always accepts a list, never a single string — callers control
  batch size for memory management during large ingestion runs.
- Internally we split into sub-batches of BATCH_SIZE to avoid overwhelming
  local servers (Ollama) or hitting API payload limits.
"""

from abc import ABC, abstractmethod

from codebase_assistant.config import settings

# Maximum texts to embed in a single API call.
# Ollama can handle larger batches but this keeps memory predictable.
_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All implementations must produce fixed-dimension float vectors where
    cosine similarity between two vectors reflects semantic similarity
    between the original texts.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Batch of strings to embed. Must be non-empty.

        Returns:
            List of float vectors, one per input text.
            All vectors have the same length (== self.dimension()).
        """
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension.

        ChromaDB needs this when creating a collection with a custom
        embedding function. For most models this is fixed (e.g., 768 for
        nomic-embed-text, 1536 for text-embedding-3-small).
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider+model name for logging."""
        ...


# ---------------------------------------------------------------------------
# Ollama implementation
# ---------------------------------------------------------------------------

class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings via Ollama's OpenAI-compatible /v1/embeddings endpoint.

    Uses the same trick as OllamaProvider (chat): the openai SDK pointed
    at Ollama's base URL with a dummy API key. The only difference is we
    call client.embeddings.create() instead of client.chat.completions.create().

    Recommended model: nomic-embed-text
    - 768 dimensions
    - Trained on both code and natural language (good for cross-modal search)
    - Fast on CPU, ~274 MB download
    - Run: ollama pull nomic-embed-text
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        from openai import OpenAI

        self._model = model or settings.embedding_model
        self._base_url = base_url or settings.embedding_base_url
        self._client = OpenAI(base_url=self._base_url, api_key="ollama")
        self._dim: int | None = None  # lazily detected on first embed call

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except Exception as e:
                error_msg = str(e).lower()
                if "connection" in error_msg or "refused" in error_msg:
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self._base_url}. "
                        "Is Ollama running? Start it with: ollama serve"
                    )
                if "not found" in error_msg or "pull" in error_msg:
                    raise RuntimeError(
                        f"Embedding model '{self._model}' not found in Ollama. "
                        f"Run: ollama pull {self._model}"
                    )
                raise RuntimeError(f"Ollama embedding error: {e}")

            # response.data is sorted by index, matching our batch order
            all_vectors.extend(item.embedding for item in response.data)

        # Cache dimension from first real call
        if self._dim is None and all_vectors:
            self._dim = len(all_vectors[0])

        return all_vectors

    def dimension(self) -> int:
        if self._dim is None:
            # Embed a single probe text to detect dimension
            self.embed(["dimension probe"])
        return self._dim  # type: ignore[return-value]

    def name(self) -> str:
        return f"Ollama ({self._model})"


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via OpenAI's /v1/embeddings API.

    Recommended model: text-embedding-3-small
    - 1536 dimensions (can be reduced via `dimensions` param)
    - Strong multilingual and code understanding
    - ~$0.02 per million tokens — cheap but not free

    Uses the same API key as the chat provider (OPENAI_API_KEY).
    """

    # Known dimensions so we don't need a probe call
    _KNOWN_DIMS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._dim: int | None = self._KNOWN_DIMS.get(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except Exception as e:
                raise RuntimeError(f"OpenAI embedding error: {e}")

            all_vectors.extend(item.embedding for item in response.data)

        if self._dim is None and all_vectors:
            self._dim = len(all_vectors[0])

        return all_vectors

    def dimension(self) -> int:
        if self._dim is None:
            self.embed(["dimension probe"])
        return self._dim  # type: ignore[return-value]

    def name(self) -> str:
        return f"OpenAI ({self._model})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedding_provider() -> EmbeddingProvider:
    """Create an EmbeddingProvider based on the configured embedding_model.

    Resolution logic:
    - If embedding_model starts with "text-embedding-" → OpenAI provider
      (requires OPENAI_API_KEY to be set)
    - Otherwise → Ollama provider (default, free, local)

    This mirrors the chat factory pattern: config drives the decision,
    callers never need to know which provider they're using.
    """
    model = settings.embedding_model

    if model.startswith("text-embedding-"):
        key = settings.openai_api_key
        if not key:
            raise RuntimeError(
                f"embedding_model is '{model}' but OPENAI_API_KEY is not set. "
                "Add it to .env or switch to an Ollama model (e.g., nomic-embed-text)."
            )
        return OpenAIEmbeddingProvider(api_key=key.get_secret_value(), model=model)

    return OllamaEmbeddingProvider(model=model)
