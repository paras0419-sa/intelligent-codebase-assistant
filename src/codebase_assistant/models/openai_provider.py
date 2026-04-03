import openai

from codebase_assistant.models.base import Message, ModelProvider, ModelResponse


class OpenAIProvider(ModelProvider):
    """OpenAI LLM provider.

    Key OpenAI API differences from Anthropic:
    - system prompt goes as the FIRST message with role="system"
      (not a separate top-level parameter)
    - No strict alternation requirement — multiple user messages in a
      row are fine.
    - Token usage fields: prompt_tokens / completion_tokens (not
      input_tokens / output_tokens like Anthropic).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def chat(
        self, messages: list[Message], system: str | None = None
    ) -> ModelResponse:
        # OpenAI expects system as the first message in the list.
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})

        api_messages.extend(
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
            )
        except openai.AuthenticationError:
            raise RuntimeError(
                "OpenAI API key is invalid. Check OPENAI_API_KEY in .env"
            )
        except openai.RateLimitError:
            raise RuntimeError("OpenAI rate limit hit. Wait a moment and retry.")
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI API error: {e.status_code} {e.message}")

        choice = response.choices[0]
        usage = response.usage

        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        )

    def name(self) -> str:
        return f"OpenAI ({self._model})"
