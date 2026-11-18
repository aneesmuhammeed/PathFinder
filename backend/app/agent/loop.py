"""The agent loop: one user turn → (model → tools → model …) → grounded answer, as a stream of events.

Event types (all JSON-serialisable dicts):
  conversation  {conversation_id}
  tool_call     {id, name, input}
  tool_result   {id, name, ok, summary}
  message       {text, jobs, grounding, usage, steps, stop_reason}
  error         {message, retryable}
"""

import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agent import memory, prompts
from app.agent.grounding import annotate_unverified, check_grounding
from app.agent.tools import ToolContext, execute_tool, tool_specs
from app.db.models import Conversation, Job
from app.llm.types import LLMClient, LLMError, Message, Usage
from app.search.embedder import Embedder
from app.search.service import job_to_out
from app.services import users as users_svc

log = logging.getLogger("pathfinder.agent")

REFUSAL_TEXT = "I can't help with that request. I'm happy to help with finding jobs or planning your next career step."
EXHAUSTED_TEXT = ("I wasn't able to finish that in a reasonable number of steps. "
                  "Could you rephrase or narrow the request?")
EMPTY_TEXT = "I couldn't put together an answer to that. Could you rephrase it?"


@dataclass
class AgentConfig:
    max_steps: int = 6
    max_tokens: int = 16000
    history_token_budget: int = 30000


def get_or_create_conversation(session: Session, user_id: uuid.UUID, conversation_id: uuid.UUID | None, first_text: str) -> Conversation:
    if conversation_id is not None:
        conv = session.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user_id:  # same error for both: don't reveal other users' ids
            raise users_svc.NotFound("conversation not found")
        return conv
    conv = Conversation(user_id=user_id, title=" ".join(first_text.split())[:60])
    session.add(conv)
    session.flush()
    return conv


def _job_cards(session: Session, ids: list[int], saved: set[int], notes: dict[int, dict] | None = None) -> list[dict]:
    """Cards for the UI: the job, whether it's saved, and (when a tool explained it) why it matches."""
    if not ids:
        return []
    rows = {j.id: j for j in session.scalars(select(Job).where(Job.id.in_(ids)).options(selectinload(Job.company)))}
    cards = []
    for i in ids:
        if i in rows:
            card = job_to_out(rows[i]).model_dump(mode="json")
            card["saved"] = i in saved
            card.update((notes or {}).get(i, {}))
            cards.append(card)
    return cards


def run_turn(
    session: Session, llm: LLMClient, embedder: Embedder, *, user_id: uuid.UUID, text: str,
    conversation_id: uuid.UUID | None = None, config: AgentConfig | None = None, now: datetime | None = None,
) -> Iterator[dict]:
    cfg = config or AgentConfig()
    now = now or datetime.now(timezone.utc)
    users_svc.get_user(session, user_id)
    conv = get_or_create_conversation(session, user_id, conversation_id, text)
    yield {"type": "conversation", "conversation_id": str(conv.id)}

    memory.compact_if_needed(session, llm, conv, cfg.history_token_budget)
    visible = memory.load_visible(session, conv)
    profile = users_svc.profile_signals(users_svc.get_profile(session, user_id))
    notes = users_svc.get_profile(session, user_id).summary
    ctx_block = memory.context_block_if_changed(visible, profile, notes, now.date())
    memory.append_message(session, conv, "user", ([ctx_block] if ctx_block else []) + [{"type": "text", "text": text}])
    session.commit()

    messages = memory.load_visible(session, conv)
    ctx = ToolContext(session, embedder, user_id, now)
    tools, usage, seen, repaired = tool_specs(), Usage(), set(), False
    notes: dict[int, dict] = {}  # job id -> why it matches, collected from this turn's tool results

    for step in range(1, cfg.max_steps + 1):
        try:
            resp = llm.complete(system=prompts.SYSTEM_PROMPT, messages=messages, tools=tools, max_tokens=cfg.max_tokens)
        except LLMError as exc:
            yield {"type": "error", "message": str(exc), "retryable": exc.retryable}
            return
        usage += resp.usage

        if resp.stop_reason == "refusal":  # check before reading content
            log.warning("model refusal", extra={"category": resp.detail})
            memory.append_message(session, conv, "assistant", [{"type": "text", "text": REFUSAL_TEXT}])
            session.commit()
            yield _final(session, REFUSAL_TEXT, [], [], usage, step, "refusal")
            return
        if resp.stop_reason == "max_tokens":
            yield {"type": "error", "message": "The answer got too long to finish. Try a more specific question.", "retryable": True}
            return

        uses = resp.tool_uses
        if uses:
            results = []
            for tu in uses:
                yield {"type": "tool_call", "id": tu["id"], "name": tu["name"], "input": tu["input"]}
                out = execute_tool(ctx, tu["name"], tu["input"])
                seen |= out.job_ids
                notes.update(out.job_notes)
                yield {"type": "tool_result", "id": tu["id"], "name": tu["name"], "ok": not out.is_error, "summary": out.summary}
                block = {"type": "tool_result", "tool_use_id": tu["id"], "content": out.content}
                if out.is_error:
                    block["is_error"] = True
                results.append(block)
            assistant, tool_msg = Message("assistant", resp.content), Message("user", results)
            memory.append_message(session, conv, "assistant", assistant.content)  # both persisted together:
            memory.append_message(session, conv, "user", tool_msg.content)  # history never holds a dangling tool_use
            session.commit()
            messages += [assistant, tool_msg]
            continue

        answer = resp.text or EMPTY_TEXT
        grounding = check_grounding(answer, seen)
        content = resp.content if resp.text else [*resp.content, {"type": "text", "text": EMPTY_TEXT}]
        memory.append_message(session, conv, "assistant", content)
        if not grounding.ok and not repaired:
            repaired = True  # one repair attempt, appended (never rewriting the bad draft: history is append-only)
            notice = Message("user", [{"type": "text", "text": prompts.repair_notice(grounding.unknown)}])
            memory.append_message(session, conv, "user", notice.content)
            session.commit()
            messages += [Message("assistant", content), notice]
            log.warning("ungrounded job ids, retrying", extra={"unknown": grounding.unknown})
            continue
        session.commit()
        if not grounding.ok:
            answer = annotate_unverified(answer, grounding.unknown)
        saved = {j.id for j in users_svc.feedback_jobs(session, user_id, "saved")}
        cited_ok = [i for i in grounding.cited if i in seen]
        yield _final(session, answer, _job_cards(session, cited_ok, saved, notes), grounding.unknown, usage, step, resp.stop_reason,
                     grounding.cited)
        return

    memory.append_message(session, conv, "assistant", [{"type": "text", "text": EXHAUSTED_TEXT}])
    session.commit()
    yield _final(session, EXHAUSTED_TEXT, [], [], usage, cfg.max_steps, "max_steps")


def _final(session, text, jobs, unknown, usage: Usage, steps, stop_reason, cited=None) -> dict:
    return {
        "type": "message", "text": text, "jobs": jobs, "steps": steps, "stop_reason": stop_reason,
        "grounding": {"cited": cited or [], "unverified": unknown},
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
    }
