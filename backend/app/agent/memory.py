"""Conversation memory.

Short-term: the message history (JSONB content blocks, stored exactly as exchanged).
Long-term: the user's saved profile (survives across conversations), surfaced to the model as a
`<user_context>` block attached to a user turn *when it changes*.

Two constraints shape this module (both come from how Claude's preserved thinking and prompt caching work):

1. History is APPEND-ONLY. We never edit, reorder or delete earlier turns, never rebuild the system prompt,
   and never re-render earlier context. Rebuilding anything earlier in the prefix invalidates the thinking
   blocks of every later turn (and the cache). That is why dynamic context rides inside persisted user turns.
2. Compaction is "simple compaction": when history gets too long it is replaced wholesale by one summary,
   and nothing from before the summary is replayed (a keep-the-last-N-turns scheme would replay thinking
   blocks created under a different prefix). Compaction only happens at a turn boundary, never mid tool round.
"""

import hashlib
import json
import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import prompts
from app.db.models import Conversation
from app.db.models import Message as DBMessage
from app.llm.types import LLMClient, LLMError, Message
from app.search.schema import ProfileSignals

log = logging.getLogger("pathfinder.agent.memory")


# ---------------------------------------------------------------- persistence

def append_message(session: Session, conv: Conversation, role: str, content: list[dict]) -> DBMessage:
    msg = DBMessage(conversation_id=conv.id, role=role, content=content)
    session.add(msg)
    session.flush()
    return msg


def load_visible(session: Session, conv: Conversation) -> list[Message]:
    """The messages the model should see: everything after the last compaction, with the summary
    prepended to the first of them (deterministic, so the prefix is identical on every request)."""
    stmt = select(DBMessage).where(DBMessage.conversation_id == conv.id).order_by(DBMessage.id)
    if conv.summarized_through_id:
        stmt = stmt.where(DBMessage.id > conv.summarized_through_id)
    msgs = [Message(m.role, list(m.content)) for m in session.scalars(stmt)]
    if conv.summary and msgs and msgs[0].role == "user":
        block = {"type": "text", "text": f"{prompts.SUMMARY_TAG}\n{conv.summary}\n</conversation_summary>"}
        msgs[0] = Message("user", [block, *msgs[0].content])
    return msgs


# ---------------------------------------------------------------- profile context

def context_rev(profile: ProfileSignals, summary: str | None, today: date) -> str:
    payload = json.dumps([profile.model_dump(), summary, today.isoformat()], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def render_context(profile: ProfileSignals, summary: str | None, today: date, rev: str) -> str:
    def line(label: str, value) -> str | None:
        if value in (None, "", []):
            return None
        return f"- {label}: {', '.join(value) if isinstance(value, list) else value}"

    rows = [
        line("skills", profile.skills), line("target roles", profile.target_roles), line("level", profile.seniority),
        line("wants", profile.job_type), line("work mode", profile.work_mode),
        line("preferred locations", profile.preferred_locations),
        line("minimum salary (INR/yr)", profile.min_salary), line("notes", summary),
    ]
    body = "\n".join(r for r in rows if r) or "- nothing saved yet"
    return f'{prompts.CONTEXT_TAG} rev="{rev}">\nToday: {today.isoformat()}\nSaved profile:\n{body}\n</user_context>'


def latest_context_rev(messages: list[Message]) -> str | None:
    """Revision of the newest <user_context> block still visible to the model."""
    for m in reversed(messages):
        if m.role != "user":
            continue
        for b in m.content:
            if b.get("type") == "text" and b["text"].startswith(prompts.CONTEXT_TAG):
                start = b["text"].find('rev="')
                return b["text"][start + 5 : start + 15] if start >= 0 else None
    return None


def context_block_if_changed(visible: list[Message], profile: ProfileSignals, summary: str | None, today: date) -> dict | None:
    rev = context_rev(profile, summary, today)
    if latest_context_rev(visible) == rev:
        return None
    return {"type": "text", "text": render_context(profile, summary, today, rev)}


# ---------------------------------------------------------------- compaction

def estimate_tokens(messages: list[Message]) -> int:
    """Rough (chars/4). Thinking blocks are excluded: their signatures are large but aren't prompt text."""
    chars = 0
    for m in messages:
        for b in m.content:
            t = b.get("type")
            if t == "text":
                chars += len(b["text"])
            elif t == "tool_use":
                chars += len(json.dumps(b["input"])) + len(b["name"])
            elif t == "tool_result":
                chars += len(b["content"])
    return chars // 4


def _transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        for b in m.content:
            t = b.get("type")
            if t == "text":
                text = b["text"]
                if text.startswith((prompts.CONTEXT_TAG, prompts.NOTICE_TAG)):
                    continue
                lines.append(f"{'USER' if m.role == 'user' else 'ASSISTANT'}: {text}")
            elif t == "tool_use":
                lines.append(f"[assistant called {b['name']} {json.dumps(b['input'], ensure_ascii=False)[:200]}]")
            elif t == "tool_result":
                lines.append(f"[tool result: {b['content'][:600]}]")
    return "\n".join(lines)


def compact_if_needed(session: Session, llm: LLMClient, conv: Conversation, budget_tokens: int) -> bool:
    """Summarise the whole visible history into `conv.summary` when it exceeds the budget.
    Best effort: on any failure the conversation simply continues uncompacted."""
    visible = load_visible(session, conv)
    if not visible or estimate_tokens(visible) <= budget_tokens:
        return False
    last_id = session.scalar(select(DBMessage.id).where(DBMessage.conversation_id == conv.id).order_by(DBMessage.id.desc()).limit(1))
    prior = f"Earlier summary:\n{conv.summary}\n\nNew messages:\n" if conv.summary else ""
    try:
        resp = llm.complete(
            system=prompts.SUMMARY_SYSTEM,
            messages=[Message("user", [{"type": "text", "text": prior + _transcript(visible)}])],
            tools=[], max_tokens=2000,
        )
    except LLMError:
        log.warning("compaction skipped: summariser failed", exc_info=True)
        return False
    summary = resp.text.strip()
    if not summary:
        return False
    conv.summary, conv.summarized_through_id = summary, last_id
    session.commit()
    log.info("conversation compacted", extra={"conversation_id": str(conv.id), "summary_chars": len(summary)})
    return True


# ---------------------------------------------------------------- UI-facing view

def public_history(db_messages: list[DBMessage]) -> list[dict]:
    """What a chat UI should render: plain user/assistant text. Hides tool plumbing, injected context,
    and any assistant answer that was superseded by a grounding-repair retry."""
    out: list[dict] = []
    rows = list(db_messages)
    for i, m in enumerate(rows):
        texts = [b["text"] for b in m.content if b.get("type") == "text"]
        if m.role == "user":
            texts = [t for t in texts if not t.startswith((prompts.CONTEXT_TAG, prompts.NOTICE_TAG))]
        if not texts:
            continue
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if m.role == "assistant" and nxt and any(
            b.get("type") == "text" and b["text"].startswith(prompts.NOTICE_TAG) for b in nxt.content
        ):
            continue  # superseded draft
        out.append({"id": m.id, "role": m.role, "text": "\n\n".join(texts), "created_at": m.created_at})
    return out
