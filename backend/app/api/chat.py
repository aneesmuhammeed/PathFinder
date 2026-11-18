import json
import logging
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.agent.loop import AgentConfig, get_or_create_conversation, run_turn
from app.agent.memory import public_history
from app.api.deps import SessionFactory, get_embedder_dep, get_llm_dep, get_session_factory
from app.core.config import get_settings
from app.core.ratelimit import limit_chat
from app.db.models import Conversation, Message
from app.llm import LLMClient
from app.search.embedder import Embedder
from app.services import users as users_svc

log = logging.getLogger("pathfinder.chat")
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    user_id: uuid.UUID
    conversation_id: uuid.UUID | None = Field(None, description="omit to start a new conversation")
    message: str = Field(min_length=1, max_length=20000)
    stream: bool = True

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for ev in events:
        yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat")
def chat(
    req: ChatRequest,
    request: Request,
    factory: SessionFactory = Depends(get_session_factory),
    llm: LLMClient = Depends(get_llm_dep),
    embedder: Embedder = Depends(get_embedder_dep),
):
    """Run one agent turn. `stream=true` (default) answers as server-sent events:
    conversation → tool_call/tool_result (zero or more) → message | error."""
    s = get_settings()
    if len(req.message) > s.max_message_chars:
        raise HTTPException(422, f"message too long (max {s.max_message_chars} characters)")
    with factory() as session:  # validate up front so bad ids are a clean 404, not a broken stream
        try:
            users_svc.get_user(session, req.user_id)
            if req.conversation_id:
                get_or_create_conversation(session, req.user_id, req.conversation_id, "")
        except users_svc.NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
    limit_chat(request, str(req.user_id))  # after validation: junk requests shouldn't burn the daily budget

    cfg = AgentConfig(s.agent_max_steps, s.llm_max_tokens, s.agent_history_token_budget)

    def events() -> Iterator[dict]:
        with factory() as session:
            try:
                yield from run_turn(session, llm, embedder, user_id=req.user_id, text=req.message,
                                    conversation_id=req.conversation_id, config=cfg)
            except Exception:  # noqa: BLE001 - never leave a stream hanging without a terminal event
                log.exception("agent turn crashed")
                session.rollback()
                yield {"type": "error", "message": "Something went wrong. Please try again.", "retryable": True}

    if req.stream:
        return StreamingResponse(_sse(events()), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    collected = list(events())
    final = next((e for e in collected if e["type"] in ("message", "error")), None)
    if final is None or final["type"] == "error":
        raise HTTPException(502 if final else 500, final["message"] if final else "no response")
    conv = next(e["conversation_id"] for e in collected if e["type"] == "conversation")
    return {"conversation_id": conv, "message": final,
            "tool_calls": [{"name": e["name"], "input": e["input"]} for e in collected if e["type"] == "tool_call"]}


@router.get("/users/{user_id}/conversations")
def list_conversations(user_id: uuid.UUID, factory: SessionFactory = Depends(get_session_factory)) -> list[dict]:
    with factory() as session:
        try:
            users_svc.get_user(session, user_id)
        except users_svc.NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        rows = session.scalars(
            select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()).limit(50)
        )
        return [{"id": str(c.id), "title": c.title, "updated_at": c.updated_at} for c in rows]


@router.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: uuid.UUID, user_id: uuid.UUID, factory: SessionFactory = Depends(get_session_factory)) -> list[dict]:
    """Chat-UI view of a conversation (tool plumbing and injected context hidden). `user_id` must own it."""
    with factory() as session:
        conv = session.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user_id:
            raise HTTPException(404, "conversation not found")
        rows = session.scalars(select(Message).where(Message.conversation_id == conv.id).order_by(Message.id))
        return public_history(list(rows))
