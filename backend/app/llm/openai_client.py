"""OpenAI Chat Completions protocol client. Also serves Gemini (via Google's OpenAI-compatible endpoint)
and any local server that speaks the protocol (Ollama, vLLM): just a different base_url / key / model.

`to_openai_messages` / `from_openai_message` are pure functions so the translation is unit-testable.
"""

import json
import logging
from typing import Any

from app.llm.types import LLMError, LLMResponse, Message, StopReason, ToolSpec, Usage

log = logging.getLogger("pathfinder.llm.openai")

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_FINISH_MAP: dict[str, StopReason] = {
    "stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use",
    "length": "max_tokens", "content_filter": "refusal",
}


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "assistant":
            text = "".join(b.get("text", "") for b in m.content if b["type"] == "text")
            calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)}}
                for b in m.content if b["type"] == "tool_use"
            ]  # `thinking` blocks are Anthropic-specific and silently dropped
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            if text or calls:
                out.append(msg)
        else:
            for b in m.content:  # tool results become their own "tool" messages, in order
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
            text = "\n\n".join(b["text"] for b in m.content if b["type"] == "text")
            if text:
                out.append({"role": "user", "content": text})
    return out


def from_openai_message(msg: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        fn = call["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError:
            args = {"_unparsed_arguments": str(fn.get("arguments"))[:500]}  # fails validation → model retries
        blocks.append({"type": "tool_use", "id": call["id"], "name": fn["name"], "input": args})
    return blocks


class OpenAICompatClient:
    def __init__(self, *, name: str = "openai", model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, client: Any = None, timeout: float | None = None):
        self.name = name
        self.model = model or OPENAI_DEFAULT_MODEL
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("openai not installed: pip install -r requirements-llm.txt") from exc
            client = openai.OpenAI(api_key=api_key, base_url=base_url, **({"timeout": timeout} if timeout else {}))
        self._client = client

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 16000) -> LLMResponse:
        import openai

        params: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            params["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]
        try:
            resp = self._client.chat.completions.create(**params)
        except openai.RateLimitError as exc:
            raise LLMError("The AI service is rate-limited right now. Please retry shortly.", retryable=True) from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the AI service.", retryable=True) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"AI service error ({exc.status_code})", retryable=exc.status_code >= 500) from exc

        choice = resp.choices[0]
        content = from_openai_message(choice.message.model_dump(exclude_none=True))
        stop = _FINISH_MAP.get(choice.finish_reason or "", "other")
        if any(b["type"] == "tool_use" for b in content):
            stop = "tool_use"  # some compatible servers report "stop" even with tool calls
        usage = Usage(getattr(resp.usage, "prompt_tokens", 0) or 0, getattr(resp.usage, "completion_tokens", 0) or 0)
        return LLMResponse(content=content, stop_reason=stop, usage=usage, model=resp.model or self.model)
