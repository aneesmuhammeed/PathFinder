"""Shared builders for search tests: tiny hand-made corpora ingested through the real pipeline."""

import copy
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Job
from app.ingest.pipeline import run_ingestion
from app.ingest.schema import NormalizedJob
from app.llm.types import LLMResponse, Usage
from app.search.embedder import HashEmbedder

NOW = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)
EMB = HashEmbedder(384)


class ListSource:
    name, snapshot, min_interval = "t", True, None

    def __init__(self, jobs):
        self.jobs = jobs

    def fetch(self):
        return list(self.jobs)


def nj(key: str, title: str, skills: list[str], *, days_ago: int | None = 3, desc: str | None = None, **kw) -> NormalizedJob:
    base = dict(
        source="t", external_id=key, company=kw.pop("company", f"Co-{key}"), title=title,
        description=desc or f"{title}. We build services and ship software. Skills: {', '.join(skills)}.",
        skills=skills, location="Bengaluru", country="India",
        posted_at=None if days_ago is None else NOW - timedelta(days=days_ago),
    )
    base.update(kw)
    return NormalizedJob(**base)


def load(db: Session, jobs: list[NormalizedJob], embed: bool = True) -> dict[str, int]:
    """Ingest `jobs`; returns external_id -> job id. embed=False leaves vectors empty."""
    embedder = EMB if embed else _NoEmbed()
    run = run_ingestion(db, ListSource(jobs), embedder, now=NOW, force=True)
    assert run.status == "success", run.error
    return {ext_id: job_id for ext_id, job_id in db.execute(select(Job.external_id, Job.id).where(Job.source == "t"))}


class ScriptedLLM:
    """Plays back canned responses and records every request it receives (for append-only / stability checks).

    Each script item is an LLMResponse, an Exception to raise, or a callable(system, messages, tools) -> LLMResponse.
    """

    name, model = "scripted", "scripted-1"

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, *, system, messages, tools, max_tokens=16000):
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": list(tools)})
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(system, messages, tools) if callable(item) else item


def say(text: str, **kw) -> LLMResponse:
    return LLMResponse([{"type": "text", "text": text}], "end_turn", Usage(100, 20), "scripted-1", **kw)


def call(*calls, thinking: bool = False) -> LLMResponse:
    """A tool_use response: call(("search_jobs", {...}), ("get_profile", {}))."""
    blocks = [{"type": "thinking", "thinking": "", "signature": "sig-abc123"}] if thinking else []
    blocks += [{"type": "tool_use", "id": f"toolu_{i}", "name": n, "input": a} for i, (n, a) in enumerate(calls)]
    return LLMResponse(blocks, "tool_use", Usage(100, 20), "scripted-1")


class _NoEmbed(HashEmbedder):
    def embed(self, texts):
        return [None for _ in texts]  # type: ignore[misc]
