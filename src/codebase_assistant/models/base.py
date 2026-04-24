from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


@dataclass
class Message:
    """A single message in a conversation.

    We define our own Message type instead of using the SDK-specific ones
    (anthropic.types.Message, openai ChatCompletionMessage) so the rest of
    our codebase never depends on a specific provider. This is the
    "anti-corruption layer" pattern — isolate external API shapes behind
    your own domain types.
    """

    role: Role
    content: str


@dataclass
class ModelResponse:
    """Standardized response from any LLM provider.

    Wrapping the raw API response lets us:
    - Track token usage uniformly across providers
    - Add metadata (model name, latency) later without changing callers
    - Keep tool-call parsing in one place per provider
    """

    content: str
    model: str
    usage: TokenUsage


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class ToolAwareResponse:
    """Response from chat_with_tools — may contain text and/or tool call blocks."""
    content: list[TextBlock | ToolUseBlock]
    stop_reason: str  # "end_turn" | "tool_use"
    model: str

    def text(self) -> str:
        return " ".join(b.text for b in self.content if isinstance(b, TextBlock))


class ModelProvider(ABC):
    """Abstract base class for LLM providers.

    Why ABC instead of Protocol?
    - ABC enforces that subclasses implement all methods at class-definition
      time (you get an error if you forget). Protocol only checks at call
      time (duck typing), which means bugs surface later.
    - We want a clear contract: every provider MUST implement chat().
    """

    @abstractmethod
    def chat(
        self, messages: list[Message], system: str | None = None
    ) -> ModelResponse:
        """Send messages to the LLM and get a response.

        Args:
            messages: Conversation history as a list of Message objects.
            system: Optional system prompt. Kept separate from messages
                    because Anthropic and OpenAI handle system prompts
                    differently (Anthropic: top-level param, OpenAI:
                    first message with role="system").
        """
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict],
        system: str | None,
        tools: list[dict],
    ) -> ToolAwareResponse:
        """Send messages with tool schemas; returns text and/or tool call blocks.

        Args:
            messages: Raw dicts in provider-agnostic format (role + content).
                      Tool result messages use {"role": "tool", ...} shape.
            system: System prompt.
            tools: List of tool definitions in OpenAI function-calling format.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for logging."""
        ...
