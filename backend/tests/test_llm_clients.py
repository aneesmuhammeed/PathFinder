"""LLM client tests. The real provider SDKs run against *fake HTTP servers* (httpx2.MockTransport), so request
construction, response parsing and error mapping are exercised without network or keys. What this cannot
prove is how the live APIs behave: that needs a real key."""

import json

import anthropic
import httpx2
import openai
import pytest

from app.core.config import Settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.factory import get_llm_client, resolve_provider
from app.llm.mock import MockLLM
from app.llm.openai_client import OpenAICompatClient, from_openai_message, to_openai_messages
from app.llm.types import LLMError, Message, ToolSpec

TOOL = ToolSpec("get_job", "Get a job", {"type": "object", "properties": {"job_id": {"type": "integer"}}, "required": ["job_id"]})
THINKING = {"type": "thinking", "thinking": "", "signature": "sig-xyz"}
HISTORY = [
    Message("user", [{"type": "text", "text": "find python jobs"}]),
    Message("assistant", [THINKING, {"type": "tool_use", "id": "toolu_1", "name": "get_job", "input": {"job_id": 7}}]),
    Message("user", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"id":7}'}]),
]


def anthropic_reply(content, stop="end_turn", **extra):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": content,
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 7, **extra}}


def make_anthropic(handler, **kw) -> tuple[AnthropicClient, list]:
    seen: list = []

    def wrapped(request):
        seen.append(json.loads(request.content))
        return handler(request)

    sdk = anthropic.Anthropic(api_key="test-key", max_retries=0,
                              http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(wrapped)))
    return AnthropicClient(client=sdk, **kw), seen


class TestAnthropicClient:
    def test_request_shape_for_opus_5_5(self):
        client, seen = make_anthropic(lambda r: httpx2.Response(200, json=anthropic_reply([{"type": "text", "text": "hi"}])))
        resp = client.complete(system="SYS", messages=HISTORY, tools=[TOOL], max_tokens=4000)
        body = seen[0]
        assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 4000 and body["system"] == "SYS"
        # Parameters Claude Opus 5.5 rejects must never be sent.
        assert not {"thinking", "temperature", "top_p", "top_k", "tool_choice"} & body.keys()
        assert body["cache_control"] == {"type": "ephemeral"} and "output_config" not in body
        assert body["tools"] == [{"name": "get_job", "description": "Get a job", "input_schema": TOOL.input_schema}]
        assert resp.text == "hi" and resp.stop_reason == "end_turn" and resp.usage.input_tokens == 11

    def test_history_including_thinking_blocks_is_replayed_verbatim(self):
        client, seen = make_anthropic(lambda r: httpx2.Response(200, json=anthropic_reply([{"type": "text", "text": "ok"}])))
        client.complete(system="S", messages=HISTORY, tools=[TOOL])
        sent = seen[0]["messages"]
        assert sent[1]["content"][0] == THINKING  # signature untouched
        assert sent[2]["content"][0]["tool_use_id"] == "toolu_1"

    def test_response_blocks_are_stored_unchanged_for_replay(self):
        reply = anthropic_reply([THINKING, {"type": "tool_use", "id": "toolu_9", "name": "get_job", "input": {"job_id": 3}}], "tool_use")
        client, _ = make_anthropic(lambda r: httpx2.Response(200, json=reply))
        resp = client.complete(system="S", messages=HISTORY[:1], tools=[TOOL])
        assert resp.stop_reason == "tool_use"
        assert resp.content[0] == THINKING  # exact dict, ready to persist
        assert resp.tool_uses[0] == {"type": "tool_use", "id": "toolu_9", "name": "get_job", "input": {"job_id": 3}}

    def test_effort_is_sent_only_when_configured(self):
        client, seen = make_anthropic(lambda r: httpx2.Response(200, json=anthropic_reply([{"type": "text", "text": "x"}])), effort="low")
        client.complete(system="S", messages=HISTORY[:1], tools=[])
        assert seen[0]["output_config"] == {"effort": "low"} and "tools" not in seen[0]

    def test_refusal_is_surfaced_not_treated_as_content(self):
        reply = anthropic_reply([], "refusal")
        reply["stop_details"] = {"type": "refusal", "category": "cyber", "explanation": "nope"}
        client, _ = make_anthropic(lambda r: httpx2.Response(200, json=reply))
        resp = client.complete(system="S", messages=HISTORY[:1], tools=[])
        assert resp.stop_reason == "refusal" and resp.detail == "cyber"

    @pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (529, True), (400, False), (401, False)])
    def test_http_errors_map_to_llm_error(self, status, retryable):
        body = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
        client, _ = make_anthropic(lambda r: httpx2.Response(status, json=body))
        with pytest.raises(LLMError) as exc:
            client.complete(system="S", messages=HISTORY[:1], tools=[])
        assert exc.value.retryable is retryable

    def test_connection_failure_is_retryable(self):
        def boom(request):
            raise httpx2.ConnectError("down")

        client, _ = make_anthropic(boom)
        with pytest.raises(LLMError) as exc:
            client.complete(system="S", messages=HISTORY[:1], tools=[])
        assert exc.value.retryable


# ---------------------------------------------------------------- OpenAI protocol (also Gemini)

class TestOpenAITranslation:
    def test_messages_translate_and_thinking_is_dropped(self):
        out = to_openai_messages("SYS", HISTORY)
        assert out[0] == {"role": "system", "content": "SYS"}
        assert out[1] == {"role": "user", "content": "find python jobs"}
        assert out[2]["role"] == "assistant" and out[2]["content"] is None
        assert out[2]["tool_calls"] == [{"id": "toolu_1", "type": "function", "function": {"name": "get_job", "arguments": '{"job_id": 7}'}}]
        assert out[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": '{"id":7}'}
        assert "thinking" not in json.dumps(out)

    def test_tool_results_come_before_trailing_user_text(self):
        m = Message("user", [{"type": "tool_result", "tool_use_id": "a", "content": "r"}, {"type": "text", "text": "thanks"}])
        out = to_openai_messages("S", [m])
        assert [x["role"] for x in out[1:]] == ["tool", "user"]

    def test_parses_tool_calls_and_survives_bad_json(self):
        blocks = from_openai_message({"content": "hi", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_job", "arguments": '{"job_id": 5}'}},
            {"id": "c2", "type": "function", "function": {"name": "get_job", "arguments": "{oops"}},
        ]})
        assert blocks[0] == {"type": "text", "text": "hi"}
        assert blocks[1]["input"] == {"job_id": 5}
        assert blocks[2]["input"] == {"_unparsed_arguments": "{oops"}  # → validation error → model retries


def make_openai(handler, **kw) -> tuple[OpenAICompatClient, list]:
    seen: list = []

    def wrapped(request):
        seen.append(json.loads(request.content))
        return handler(request)

    sdk = openai.OpenAI(api_key="k", base_url="http://fake/v1", max_retries=0,
                        http_client=httpx2.Client(transport=httpx2.MockTransport(wrapped)))
    return OpenAICompatClient(client=sdk, **kw), seen


def openai_reply(message, finish="stop"):
    return {"id": "c", "object": "chat.completion", "created": 0, "model": "gpt-x",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}}


class TestOpenAIClient:
    def test_request_and_text_response(self):
        client, seen = make_openai(lambda r: httpx2.Response(200, json=openai_reply({"content": "hello"})))
        resp = client.complete(system="SYS", messages=HISTORY[:1], tools=[TOOL], max_tokens=500)
        body = seen[0]
        assert body["model"] == "gpt-4o-mini" and body["max_completion_tokens"] == 500
        assert body["tools"][0]["function"]["name"] == "get_job" and body["tools"][0]["function"]["parameters"] == TOOL.input_schema
        assert resp.text == "hello" and resp.stop_reason == "end_turn" and resp.usage.output_tokens == 4

    def test_tool_call_response(self):
        msg = {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_job", "arguments": '{"job_id": 2}'}}]}
        client, _ = make_openai(lambda r: httpx2.Response(200, json=openai_reply(msg, "tool_calls")))
        resp = client.complete(system="S", messages=HISTORY[:1], tools=[TOOL])
        assert resp.stop_reason == "tool_use" and resp.tool_uses[0]["input"] == {"job_id": 2}

    def test_servers_that_report_stop_with_tool_calls_still_work(self):
        msg = {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_job", "arguments": "{}"}}]}
        client, _ = make_openai(lambda r: httpx2.Response(200, json=openai_reply(msg, "stop")))
        assert client.complete(system="S", messages=HISTORY[:1], tools=[TOOL]).stop_reason == "tool_use"

    @pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False)])
    def test_errors(self, status, retryable):
        client, _ = make_openai(lambda r: httpx2.Response(status, json={"error": {"message": "x", "type": "e"}}))
        with pytest.raises(LLMError) as exc:
            client.complete(system="S", messages=HISTORY[:1], tools=[])
        assert exc.value.retryable is retryable


# ---------------------------------------------------------------- factory

class TestFactory:
    @pytest.mark.parametrize("kw,expected", [
        ({}, "mock"),
        ({"anthropic_api_key": "a"}, "anthropic"),
        ({"openai_api_key": "o"}, "openai"),
        ({"gemini_api_key": "g"}, "gemini"),
        ({"anthropic_api_key": "a", "openai_api_key": "o"}, "anthropic"),  # Anthropic is the default when several are set
        ({"llm_provider": "openai", "anthropic_api_key": "a"}, "openai"),  # explicit choice wins
    ])
    def test_resolution_order(self, kw, expected):
        assert resolve_provider(Settings(**kw)) == expected

    def test_builds_each_client(self):
        assert isinstance(get_llm_client(Settings()), MockLLM)
        c = get_llm_client(Settings(anthropic_api_key="a"))
        assert c.name == "anthropic" and c.model == "claude-opus-5-5"
        assert get_llm_client(Settings(anthropic_api_key="a", llm_model="claude-sonnet-5-5", llm_effort="low")).model == "claude-sonnet-5-5"
        g = get_llm_client(Settings(gemini_api_key="g"))
        assert g.name == "gemini" and "generativelanguage.googleapis.com" in str(g._client.base_url)
        assert get_llm_client(Settings(openai_api_key="o", openai_base_url="http://localhost:11434/v1", llm_model="llama3")).model == "llama3"

    def test_unknown_provider_is_an_error(self):
        with pytest.raises(ValueError):
            get_llm_client(Settings(llm_provider="skynet"))
