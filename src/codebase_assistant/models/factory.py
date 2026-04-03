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


def create_provider(model: str | None = None) -> ModelProvider:
    """Create a ModelProvider based on the model name.

    Resolution order:
    1. Explicit `model` argument (e.g., from --model CLI flag)
    2. DEFAULT_MODEL from settings / .env
    """
    model_name = model or settings.default_model

    if model_name.startswith("claude") or model_name.startswith("anthropic"):
        return _create_claude(model_name)
    elif model_name.startswith("gpt") or model_name.startswith("o"):
        return _create_openai(model_name)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Supported prefixes: claude*, gpt*, o*"
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
