import anthropic

from codebase_assistant.models.base import Message, ModelProvider, ModelResponse


class ClaudeProvider(ModelProvider):
    """Claude LLM provider via the Anthropic SDK.

    Key Anthropic API concepts:
    - system prompt is a TOP-LEVEL parameter, not a message in the list.
      This differs from OpenAI where system is the first message.
    - messages must strictly alternate user/assistant roles.
    - The SDK raises typed exceptions (AuthenticationError, RateLimitError,
      APIStatusError) which we catch to give useful feedback.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        # The anthropic SDK auto-reads ANTHROPIC_API_KEY from env if api_key
        # is not passed. We pass it explicitly so config.py stays the single
        # source of truth for secrets.
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def chat(
        self, messages: list[Message], system: str | None = None
    ) -> ModelResponse:
        # Convert our Message objects to the dict format Anthropic expects.
        # We filter out system messages since Anthropic takes system as a
        # separate parameter.
        api_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            raise RuntimeError(
                "Anthropic API key is invalid. Check ANTHROPIC_API_KEY in .env"
            )
        except anthropic.RateLimitError:
            raise RuntimeError(
                "Anthropic rate limit hit. Wait a moment and retry."
            )
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Anthropic API error: {e.status_code} {e.message}")

        return ModelResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def name(self) -> str:
        return f"Claude ({self._model})"
