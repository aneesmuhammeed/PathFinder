import uuid

import pytest
from sqlalchemy import select

from app.agent import prompts
from app.agent.loop import EMPTY_TEXT, EXHAUSTED_TEXT, REFUSAL_TEXT, AgentConfig, run_turn
from app.agent.memory import public_history
from app.db.models import Conversation
from app.db.models import Message as DBMessage
from app.llm.types import LLMError, LLMResponse
from app.services import users as svc
from tests.helpers import EMB, NOW, ScriptedLLM, call, load, nj, say


@pytest.fixture()
def world(db):
    ids = load(db, [
        nj("py", "Python Backend Intern", ["python", "sql"], seniority="intern", job_type="internship"),
        nj("fe", "Frontend Engineer", ["react"], seniority="mid"),
    ])
    return ids, svc.create_user(db)


def turn(db, llm, user, text, conv=None, **cfg):
    return list(run_turn(db, llm, EMB, user_id=user.id, text=text, conversation_id=conv, now=NOW,
                         config=AgentConfig(**cfg) if cfg else None))


def conv_of(events):
    return uuid.UUID(events[0]["conversation_id"])


def stored(db, conv_id):
    return list(db.scalars(select(DBMessage).where(DBMessage.conversation_id == conv_id).order_by(DBMessage.id)))


def test_happy_path_events_persistence_and_profile_memory(world, db):
    ids, user = world
    llm = ScriptedLLM([
        call(("update_profile", {"skills": ["python"], "seniority": "intern"}), ("search_jobs", {"query": "python backend internship"})),
        say(f"I'd start with #{ids['py']}, Python Backend Intern at Co-py."),
    ])
    ev = turn(db, llm, user, "I'm a student who knows Python. Find me a backend internship")
    assert [e["type"] for e in ev] == ["conversation", "tool_call", "tool_result", "tool_call", "tool_result", "message"]
    assert [e["name"] for e in ev if e["type"] == "tool_call"] == ["update_profile", "search_jobs"]
    final = ev[-1]
    assert [j["id"] for j in final["jobs"]] == [ids["py"]] and final["grounding"]["unverified"] == []
    assert final["usage"] == {"input_tokens": 200, "output_tokens": 40} and final["steps"] == 2

    msgs = stored(db, conv_of(ev))
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0].content[0]["text"].startswith("<user_context") and "nothing saved yet" in msgs[0].content[0]["text"]
    assert msgs[0].content[1]["text"].startswith("I'm a student")
    uses = [b["id"] for b in msgs[1].content if b["type"] == "tool_use"]
    results = [b["tool_use_id"] for b in msgs[2].content if b["type"] == "tool_result"]
    assert uses == results  # tool calls are always persisted together with their results
    assert svc.get_profile(db, user.id).skills == ["python"] and svc.get_profile(db, user.id).seniority == "intern"
    assert db.get(Conversation, conv_of(ev)).title.startswith("I'm a student")


def test_thinking_blocks_are_persisted_and_replayed_verbatim(world, db):
    ids, user = world
    llm = ScriptedLLM([call(("get_profile", {}), thinking=True), say("Nothing saved yet.")])
    ev = turn(db, llm, user, "what do you know about me?")
    assert stored(db, conv_of(ev))[1].content[0] == {"type": "thinking", "thinking": "", "signature": "sig-abc123"}
    assert llm.calls[1]["messages"][1].content[0] == {"type": "thinking", "thinking": "", "signature": "sig-abc123"}


def test_requests_are_append_only_and_prefix_stable_across_steps_and_turns(world, db):
    """Claude's preserved thinking and the prompt cache both require each request to extend the previous one."""
    ids, user = world
    llm = ScriptedLLM([
        call(("update_profile", {"skills": ["python"]}), thinking=True), say("Noted."),            # turn 1
        call(("search_jobs", {"query": "python"}), thinking=True), say(f"#{ids['py']}"),            # turn 2
        say("Anything else?"),                                                                      # turn 3
    ])
    e1 = turn(db, llm, user, "I know Python")
    c = conv_of(e1)
    turn(db, llm, user, "show me jobs", c)
    turn(db, llm, user, "thanks", c)
    assert len(llm.calls) == 5
    assert len({x["system"] for x in llm.calls}) == 1
    assert all(x["tools"] == llm.calls[0]["tools"] for x in llm.calls)  # same tools, same order, same schemas
    for prev, cur in zip(llm.calls, llm.calls[1:], strict=False):  # consecutive pairs
        n = len(prev["messages"])
        assert cur["messages"][:n] == prev["messages"], "an earlier message changed between requests"
        assert len(cur["messages"]) > n


def test_context_block_is_only_resent_when_profile_or_date_changes(world, db):
    ids, user = world
    llm = ScriptedLLM([say("a"), say("b"), call(("update_profile", {"skills": ["sql"]})), say("c"), say("d")])
    c = conv_of(turn(db, llm, user, "hi"))
    turn(db, llm, user, "still me", c)
    turn(db, llm, user, "I know SQL", c)
    turn(db, llm, user, "ok", c)
    users = [m for m in stored(db, c) if m.role == "user" and any(b["type"] == "text" and not b["text"].startswith("<") for b in m.content)]
    has_ctx = [any(b["text"].startswith("<user_context") for b in m.content) for m in users]
    assert has_ctx == [True, False, False, True]  # 1st turn; unchanged; (profile updated *during* turn 3); new rev on turn 4
    assert "skills: sql" in next(b["text"] for b in users[3].content if b["text"].startswith("<user_context"))


def test_grounding_repair_is_appended_not_rewritten(world, db):
    ids, user = world
    llm = ScriptedLLM([
        call(("search_jobs", {"query": "python"})),
        say("Definitely apply to #999999!"),             # never returned by a tool
        say(f"Sorry, the right one is #{ids['py']}."),
    ])
    ev = turn(db, llm, user, "find jobs")
    assert ev[-1]["text"] == f"Sorry, the right one is #{ids['py']}." and ev[-1]["grounding"]["unverified"] == []
    msgs = stored(db, conv_of(ev))
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert msgs[3].content[0]["text"] == "Definitely apply to #999999!"  # draft kept: history is append-only
    assert msgs[4].content[0]["text"].startswith(prompts.NOTICE_TAG) and "#999999" in msgs[4].content[0]["text"]
    assert [m["text"] for m in public_history(msgs)] == ["find jobs", f"Sorry, the right one is #{ids['py']}."]  # UI sees none of it


def test_persistent_hallucination_is_flagged_not_trusted(world, db):
    ids, user = world
    llm = ScriptedLLM([call(("search_jobs", {"query": "python"})), say("Try #999999."), say("Really, try #999999.")])
    final = turn(db, llm, user, "find jobs")[-1]
    assert final["text"] == "Really, try #999999 (unverified)." and final["grounding"]["unverified"] == [999999]
    assert final["jobs"] == []  # no card is shown for a job we can't vouch for


def test_tool_error_is_fed_back_and_the_model_recovers(world, db):
    ids, user = world
    llm = ScriptedLLM([call(("search_jobs", {"limit": 99})), call(("search_jobs", {"query": "python", "limit": 3})), say(f"#{ids['py']}")])
    ev = turn(db, llm, user, "find jobs")
    assert [e["ok"] for e in ev if e["type"] == "tool_result"] == [False, True]
    err_block = stored(db, conv_of(ev))[2].content[0]
    assert err_block["is_error"] is True and "limit" in err_block["content"]
    assert ev[-1]["type"] == "message" and ev[-1]["steps"] == 3


def test_step_limit_stops_runaway_tool_loops(world, db):
    ids, user = world
    llm = ScriptedLLM([call(("get_profile", {}))] * 3)
    ev = turn(db, llm, user, "loop forever", max_steps=3)
    assert ev[-1]["stop_reason"] == "max_steps" and ev[-1]["text"] == EXHAUSTED_TEXT and len(llm.calls) == 3
    msgs = stored(db, conv_of(ev))  # history stays valid: every tool_use has its result
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]


def test_refusal_is_handled_before_reading_content(world, db):
    ids, user = world
    llm = ScriptedLLM([LLMResponse([], "refusal", detail="cyber")])
    ev = turn(db, llm, user, "something the classifier dislikes")
    assert ev[-1]["stop_reason"] == "refusal" and ev[-1]["text"] == REFUSAL_TEXT
    assert stored(db, conv_of(ev))[-1].content == [{"type": "text", "text": REFUSAL_TEXT}]  # never an empty assistant message


def test_provider_errors_become_error_events_and_history_stays_valid(world, db):
    ids, user = world
    llm = ScriptedLLM([LLMError("rate limited", retryable=True), say("back again")])
    ev = turn(db, llm, user, "hello")
    assert ev[-1] == {"type": "error", "message": "rate limited", "retryable": True}
    c = conv_of(ev)
    assert [m.role for m in stored(db, c)] == ["user"]
    assert turn(db, llm, user, "hello again", c)[-1]["text"] == "back again"


def test_empty_model_answer_gets_a_fallback(world, db):
    ids, user = world
    ev = turn(db, ScriptedLLM([LLMResponse([{"type": "thinking", "thinking": "", "signature": "s"}], "end_turn")]), user, "hi")
    assert ev[-1]["text"] == EMPTY_TEXT
    assert stored(db, conv_of(ev))[-1].content[-1] == {"type": "text", "text": EMPTY_TEXT}


def test_conversations_are_private_to_their_owner(world, db):
    ids, user = world
    c = conv_of(turn(db, ScriptedLLM([say("hi")]), user, "hello"))
    other = svc.create_user(db)
    with pytest.raises(svc.NotFound):
        turn(db, ScriptedLLM([say("x")]), other, "peek", c)
    with pytest.raises(svc.NotFound):
        turn(db, ScriptedLLM([say("x")]), user, "ghost", uuid.uuid4())
    with pytest.raises(svc.NotFound):
        turn(db, ScriptedLLM([say("x")]), type("U", (), {"id": uuid.uuid4()})(), "who am i")


def test_compaction_replaces_history_with_a_summary_and_replays_nothing_older(world, db):
    ids, user = world
    long_answer = "word " * 400  # ~500 tokens
    llm = ScriptedLLM([say(long_answer), say("SUMMARY: user likes python internships"), say("fresh answer")])
    c = conv_of(turn(db, llm, user, "first question", history_token_budget=100))
    last_before = stored(db, c)[-1].id
    ev = turn(db, llm, user, "second question", c, history_token_budget=100)

    conv = db.get(Conversation, c)
    assert conv.summary == "SUMMARY: user likes python internships" and conv.summarized_through_id == last_before
    assert ev[-1]["text"] == "fresh answer"
    sent = llm.calls[2]["messages"]  # call 0: turn 1, call 1: summariser, call 2: turn 2
    assert len(sent) == 1 and sent[0].role == "user"
    texts = [b["text"] for b in sent[0].content]
    assert texts[0].startswith("<conversation_summary>") and "SUMMARY: user likes python" in texts[0]
    assert texts[1].startswith("<user_context") and texts[2] == "second question"  # context re-sent after the reset
    assert long_answer not in str(sent)  # nothing from before the summary is replayed
    assert llm.calls[1]["system"] == prompts.SUMMARY_SYSTEM and not llm.calls[1]["tools"]


def test_compaction_failure_is_harmless(world, db):
    ids, user = world
    llm = ScriptedLLM([say("word " * 400), LLMError("down"), say("still works")])
    c = conv_of(turn(db, llm, user, "q1", history_token_budget=100))
    ev = turn(db, llm, user, "q2", c, history_token_budget=100)
    assert ev[-1]["text"] == "still works" and db.get(Conversation, c).summary is None
