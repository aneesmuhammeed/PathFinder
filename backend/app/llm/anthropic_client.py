"""Claude via the official Anthropic SDK (default provider).

Model-specific facts this client respects (Claude Opus 5.5 and newer):
* Thinking is always on: we never send `thinking`, `temperature`, `top_p` or `top_k` (all rejected).
* No forced `tool_choice` (rejected); the agent steers via the prompt and validates tool calls itself.
* Thinking blocks are returned as `thinking` content blocks. We hand them back to the caller untouched
  so the agent can persist and replay them verbatim (editing history invalidates their signatures).
* Always read `stop_reason` before `content`: classifier refusals come back as HTTP 200 + "refusal".
"""

import logging
from typing import Any

from app.llm.types import LLMError, LLMResponse, Message, StopReason, ToolSpec, Usage

log = logging.getLogger("pathfinder.llm.anthropic")

DEFAULT_MODEL = "claude-opus-5-5"
_STOP_MAP: dict[str, StopReason] = {
    "end_turn": "end_turn", "stop_sequence": "end_turn", "tool_use": "tool_use",
    "max_tokens": "max_tokens", "refusal": "refusal",
}


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, effort: str | None = None,
                 client: Any = None, timeout: float | None = None):
        self.model = model or DEFAULT_MODEL
        self.effort = effort
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("anthropic not installed: pip install -r requirements-llm.txt") from exc
            # key from arg, else ANTHROPIC_API_KEY / `ant auth login`; the SDK retries 429/5xx twice by default
            client = anthropic.Anthropic(api_key=api_key, **({"timeout": timeout} if timeout else {}))
        self._client = client

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 16000) -> LLMResponse:
        import anthropic

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,  # thinking counts toward this, so keep it generous
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "cache_control": {"type": "ephemeral"},  # prefix is byte-stable by design, so this pays off
        }
        if tools:
            params["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        try:
            resp = self._client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise LLMError("The AI service is rate-limited right now. Please retry shortly.", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the AI service.", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code >= 500
            log.error("anthropic error", extra={"status": exc.status_code, "request_id": getattr(exc, "request_id", None)})
            raise LLMError(f"AI service error ({exc.status_code}): {getattr(exc, 'message', exc)}", retryable=retryable) from exc

        # Store exactly what came back (thinking blocks included); drop only unset fields.
        content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        detail = None
        details = getattr(resp, "stop_details", None)
        if resp.stop_reason == "refusal" and details is not None:
            detail = getattr(details, "category", None) or "refusal"
        u = resp.usage
        return LLMResponse(
            content=content,
            stop_reason=_STOP_MAP.get(resp.stop_reason or "", "other"),
            usage=Usage(u.input_tokens, u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0),
            model=resp.model,
            detail=detail,
        )
