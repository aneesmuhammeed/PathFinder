"""/chat over HTTP (SSE and plain JSON) driven by the deterministic mock LLM and real search."""

import json
import uuid
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_embedder_dep, get_llm_dep, get_session_factory
from app.db.session import get_db
from app.llm.mock import MockLLM
from app.llm.types import LLMError
from app.main import create_app
from app.services import users as svc
from tests.helpers import EMB, ScriptedLLM, call, load, nj, say


def build(db, llm):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db  # REST routes (profile, saved jobs) see the same rolled-back session
    app.dependency_overrides[get_session_factory] = lambda: (lambda: nullcontext(db))
    app.dependency_overrides[get_llm_dep] = lambda: llm
    app.dependency_overrides[get_embedder_dep] = lambda: EMB
    return TestClient(app)


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        ev = json.loads(lines["data"])
        assert lines["event"] == ev["type"]  # SSE event name mirrors the payload type
        events.append(ev)
    return events


@pytest.fixture()
def world(db):
    ids = load(db, [
        nj("py", "Python Backend Intern", ["python", "sql", "fastapi"], seniority="intern", job_type="internship"),
        nj("go", "Go Backend Engineer", ["go", "kubernetes"], seniority="mid", location="Pune"),
        nj("fe", "Frontend Engineer", ["react", "css"], seniority="mid"),
    ])
    return ids, svc.create_user(db)


def test_sse_stream_with_mock_llm(world, db):
    ids, user = world
    api = build(db, MockLLM())
    r = api.post("/chat", json={"user_id": str(user.id), "message": "I'm a student, I know Python and SQL. Find me a backend internship"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = parse_sse(r.text)
    assert [e["type"] for e in ev] == ["conversation", "tool_call", "tool_result", "tool_call", "tool_result", "message"]
    assert [e["name"] for e in ev if e["type"] == "tool_call"] == ["update_profile", "search_jobs"]
    final = ev[-1]
    assert f"#{ids['py']}" in final["text"] and final["jobs"][0]["id"] == ids["py"] and final["grounding"]["unverified"] == []
    # What the agent heard became long-term memory:
    p = api.get(f"/users/{user.id}/profile").json()
    assert {"python", "sql"} <= set(p["skills"]) and p["job_type"] == "internship" and p["seniority"] == "intern"


def test_final_job_cards_carry_the_explanation_the_tools_produced(world, db):
    ids, user = world
    api = build(db, MockLLM())
    api.patch(f"/users/{user.id}/profile", json={"skills": ["python", "sql"], "seniority": "intern", "job_type": "internship"})
    card = api.post("/chat", json={"user_id": str(user.id), "message": "find python backend jobs", "stream": False}).json()["message"]["jobs"][0]
    assert card["id"] == ids["py"] and 0 < card["match_score"] <= 1
    assert any("listed skills match" in w or "level matches" in w for w in card["why"]) and isinstance(card["watch_out"], list)
    explained = api.post("/chat", json={"user_id": str(user.id), "message": f"why does #{ids['go']} fit me?", "stream": False}).json()["message"]["jobs"][0]
    assert explained["id"] == ids["go"] and "match_score" in explained and explained["watch_out"]  # concerns are surfaced, not hidden


def test_multi_turn_conversation_with_memory_and_actions(world, db):
    ids, user = world
    api = build(db, MockLLM())
    first = api.post("/chat", json={"user_id": str(user.id), "message": "find python backend jobs", "stream": False}).json()
    cid = first["conversation_id"]
    # It saved the role it heard ("backend"), then searched.
    assert [c["name"] for c in first["tool_calls"]] == ["update_profile", "search_jobs"]
    assert f"#{ids['py']}" in first["message"]["text"]

    saved = api.post("/chat", json={"user_id": str(user.id), "conversation_id": cid, "message": f"save #{ids['py']}", "stream": False}).json()
    assert saved["message"]["text"] == f"Saved #{ids['py']}." and saved["conversation_id"] == cid
    assert [j["id"] for j in api.get(f"/users/{user.id}/saved").json()] == [ids["py"]]

    listing = api.post("/chat", json={"user_id": str(user.id), "conversation_id": cid, "message": "show my saved jobs", "stream": False}).json()
    assert f"#{ids['py']}" in listing["message"]["text"]

    why = api.post("/chat", json={"user_id": str(user.id), "conversation_id": cid, "message": f"why does #{ids['go']} fit me?", "stream": False}).json()
    assert why["message"]["text"].startswith(f"Fit for #{ids['go']}")

    # The UI view hides tool plumbing and injected context.
    hist = api.get(f"/conversations/{cid}/messages", params={"user_id": str(user.id)}).json()
    assert [h["role"] for h in hist] == ["user", "assistant"] * 4
    assert hist[0]["text"] == "find python backend jobs" and "<user_context" not in json.dumps(hist)
    convs = api.get(f"/users/{user.id}/conversations").json()
    assert [c["id"] for c in convs] == [cid] and convs[0]["title"] == "find python backend jobs"


def test_compare_and_dismiss_via_chat(world, db):
    ids, user = world
    api = build(db, MockLLM())
    say_ = lambda m: api.post("/chat", json={"user_id": str(user.id), "message": m, "stream": False}).json()["message"]["text"]  # noqa: E731
    assert "Comparison" in say_(f"compare #{ids['py']} and #{ids['go']}")
    assert say_(f"dismiss #{ids['fe']}").startswith("Dismissed")
    assert f"#{ids['fe']}" not in say_("find me frontend jobs")  # dismissed jobs stay hidden


def test_validation_and_not_found(world, db):
    ids, user = world
    api = build(db, MockLLM())
    ok = {"user_id": str(user.id), "message": "hi"}
    assert api.post("/chat", json={**ok, "user_id": str(uuid.uuid4())}).status_code == 404
    assert api.post("/chat", json={**ok, "conversation_id": str(uuid.uuid4())}).status_code == 404
    assert api.post("/chat", json={**ok, "message": "   "}).status_code == 422
    assert api.post("/chat", json={**ok, "message": "x" * 4001}).status_code == 422
    assert api.post("/chat", json={"message": "hi"}).status_code == 422
    other = svc.create_user(db)
    cid = api.post("/chat", json={**ok, "stream": False}).json()["conversation_id"]
    assert api.post("/chat", json={"user_id": str(other.id), "conversation_id": cid, "message": "hi"}).status_code == 404
    assert api.get(f"/conversations/{cid}/messages", params={"user_id": str(other.id)}).status_code == 404
    assert api.get(f"/users/{uuid.uuid4()}/conversations").status_code == 404


def test_provider_error_is_a_terminal_sse_event_and_a_502_without_streaming(world, db):
    ids, user = world
    api = build(db, ScriptedLLM([LLMError("rate limited", retryable=True), LLMError("rate limited", retryable=True)]))
    body = {"user_id": str(user.id), "message": "hi"}
    ev = parse_sse(api.post("/chat", json=body).text)
    assert ev[-1] == {"type": "error", "message": "rate limited", "retryable": True}
    r = api.post("/chat", json={**body, "stream": False})
    assert r.status_code == 502 and "rate limited" in r.json()["detail"]


def test_unexpected_crash_never_leaves_a_stream_hanging_or_leaks_details(world, db):
    ids, user = world

    class Exploding:
        name, model = "x", "x"

        def complete(self, **kw):
            raise RuntimeError("secret internal detail")

    api = build(db, Exploding())
    r = api.post("/chat", json={"user_id": str(user.id), "message": "hi"})
    ev = parse_sse(r.text)
    assert ev[-1]["type"] == "error" and "secret" not in r.text and ev[-1]["retryable"]


def test_stream_emits_tool_events_before_the_answer(world, db):
    ids, user = world
    api = build(db, ScriptedLLM([call(("search_jobs", {"query": "react"})), say("done")]))
    ev = parse_sse(api.post("/chat", json={"user_id": str(user.id), "message": "react jobs"}).text)
    assert [e["type"] for e in ev] == ["conversation", "tool_call", "tool_result", "message"]
    assert ev[2]["ok"] is True and ev[2]["summary"].endswith("found")
