"""Factory for creating ModelProvider instances.

Why a factory instead of just instantiating providers directly?
- The CLI/agent code shouldn't know which provider is configured.
  It just calls `create_provider()` and gets back a ModelProvider.
- Validation (is the API key set?) happens in one place.
- Adding a new provider (Ollama, Gemini) means adding one elif branch
  here — no changes to agent code.
"""

from codebase_assistant.config import settings
from codebase_assistant.models.base import ModelProvider

# Explicit prefixes for each provider. Prefix-matching on single letters
# like "o" is too greedy — "ollama/llama3" would wrongly route to OpenAI.
_ANTHROPIC_PREFIXES = ("claude", "anthropic")
_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4")
_GEMINI_PREFIXES = ("gemini",)
_OLLAMA_PREFIXES = ("ollama/", "mistral", "llama", "qwen")


def create_provider(model: str | None = None) -> ModelProvider:
    """Create a ModelProvider based on the model name.

    Resolution order:
    1. Explicit `model` argument (e.g., from --model CLI flag)
    2. DEFAULT_MODEL from settings / .env
    """
    model_name = model or settings.default_model

    if model_name.startswith(_ANTHROPIC_PREFIXES):
        return _create_claude(model_name)
    elif model_name.startswith(_OPENAI_PREFIXES):
        return _create_openai(model_name)
    elif model_name.startswith(_GEMINI_PREFIXES):
        return _create_gemini(model_name)
    elif model_name.startswith(_OLLAMA_PREFIXES):
        return _create_ollama(model_name)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported prefixes: {_ANTHROPIC_PREFIXES + _OPENAI_PREFIXES + _GEMINI_PREFIXES + _OLLAMA_PREFIXES}"
        )


def _create_claude(model: str) -> ModelProvider:
    from codebase_assistant.models.claude import ClaudeProvider

    key = settings.anthropic_api_key
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export it."
        )
    return ClaudeProvider(api_key=key.get_secret_value(), model=model)


def _create_openai(model: str) -> ModelProvider:
    from codebase_assistant.models.openai_provider import OpenAIProvider

    key = settings.openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env or export it."
        )
    return OpenAIProvider(api_key=key.get_secret_value(), model=model)


def _create_gemini(model: str) -> ModelProvider:
    from codebase_assistant.models.gemini_provider import GeminiProvider

    key = settings.gemini_api_key
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env or export it."
        )
    return GeminiProvider(api_key=key.get_secret_value(), model=model)


def _create_ollama(model: str) -> ModelProvider:
    from codebase_assistant.models.ollama_provider import OllamaProvider

    # Strip "ollama/" prefix — it's a routing hint, not part of the model name.
    # e.g., "ollama/mistral:latest" → "mistral:latest"
    if model.startswith("ollama/"):
        model = model[len("ollama/"):]
    return OllamaProvider(model=model)
