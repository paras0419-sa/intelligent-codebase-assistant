from google import genai
from google.genai import types

from codebase_assistant.config import settings
from codebase_assistant.models.base import Message, ModelProvider, ModelResponse


class GeminiProvider(ModelProvider):
    """Google Gemini LLM provider via the google-genai SDK.

    Key Gemini API differences from Anthropic/OpenAI:
    - Uses google.genai.Client, not a REST-style client.
    - System prompt is passed via GenerateContentConfig, similar to
      Anthropic's top-level param approach.
    - Role mapping: Gemini uses "user" and "model" (not "assistant").
    - Token usage fields: prompt_token_count / candidates_token_count.
    - Free tier: 15 requests/min on Gemini 2.0 Flash — enough for dev.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def chat(
        self, messages: list[Message], system: str | None = None
    ) -> ModelResponse:
        # Gemini uses "model" instead of "assistant" for the AI role.
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else m.role,
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]

        config = types.GenerateContentConfig(
            max_output_tokens=settings.max_tokens,
        )
        if system:
            config.system_instruction = system

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authenticate" in error_msg:
                raise RuntimeError(
                    "Gemini API key is invalid. Check GEMINI_API_KEY in .env"
                )
            if "rate" in error_msg or "quota" in error_msg:
                raise RuntimeError(
                    "Gemini rate limit hit. Free tier allows 15 RPM."
                )
            raise RuntimeError(f"Gemini API error: {e}")

        text = response.text or ""
        usage_meta = response.usage_metadata

        return ModelResponse(
            content=text,
            model=self._model,
            usage={
                "input_tokens": usage_meta.prompt_token_count if usage_meta else 0,
                "output_tokens": usage_meta.candidates_token_count if usage_meta else 0,
            },
        )

    def name(self) -> str:
        return f"Gemini ({self._model})"
