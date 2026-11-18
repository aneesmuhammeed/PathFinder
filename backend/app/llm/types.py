"""Provider-neutral conversation types.

Content blocks are plain JSON dicts in the Anthropic shape, because that is also what we persist:

    {"type": "text", "text": "..."}
    {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    {"type": "tool_result", "tool_use_id": "...", "content": "...", "is_error": true?}
    {"type": "thinking", ...}   # opaque provider block: stored and replayed UNCHANGED, never edited

Why dicts: Claude's thinking blocks carry signatures bound to the conversation prefix. Replaying history
that we re-serialised or edited can invalidate them, so what the API returned is what we store and send back.
Providers that can't use a block type (e.g. OpenAI and `thinking`) drop it on the way out.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant"]
Block = dict[str, Any]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


@dataclass
class Message:
    role: Role
    content: list[Block]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        return self


@dataclass
class LLMResponse:
    content: list[Block]
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    detail: str | None = None  # e.g. refusal category

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[Block]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class LLMError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMClient(Protocol):
    name: str
    model: str

    def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 16000
    ) -> LLMResponse: ...
