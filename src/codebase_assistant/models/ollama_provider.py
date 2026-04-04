"""Ollama LLM provider — runs models locally via Ollama's OpenAI-compatible API.

Key insight: Ollama exposes an OpenAI-compatible endpoint at
http://localhost:11434/v1, so we reuse the `openai` SDK with a custom
base_url. This means:
- No new dependency needed
- Same message format, role names, and response shape as OpenAI
- System prompt handling identical to OpenAI (first message with role="system")

The difference from OpenAIProvider:
- No API key required (local server)
- base_url points to localhost instead of api.openai.com
- Token usage may not always be populated (depends on model/version)
"""

from openai import OpenAI

from codebase_assistant.config import settings
from codebase_assistant.models.base import Message, ModelProvider, ModelResponse


class OllamaProvider(ModelProvider):
    """Local LLM provider via Ollama's OpenAI-compatible API."""

    def __init__(self, model: str = "mistral:latest", base_url: str | None = None):
        self._model = model
        self._base_url = base_url or settings.ollama_base_url
        # Ollama doesn't need a real API key, but the OpenAI SDK requires one.
        # "ollama" is a conventional dummy value.
        self._client = OpenAI(base_url=self._base_url, api_key="ollama")

    def chat(
        self, messages: list[Message], system: str | None = None
    ) -> ModelResponse:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        oai_messages.extend(
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=oai_messages,
                max_tokens=settings.max_tokens,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Is Ollama running? Start it with: ollama serve"
                )
            raise RuntimeError(f"Ollama API error: {e}")

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage

        return ModelResponse(
            content=text,
            model=self._model,
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        )

    def name(self) -> str:
        return f"Ollama ({self._model})"
