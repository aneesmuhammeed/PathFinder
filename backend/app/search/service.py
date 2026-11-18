"""Search orchestration: resolve profile → retrieve → (rerank) → explainable results."""

import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Job
from app.ingest.skills import extract_skills
from app.search.embedder import Embedder
from app.search.hybrid import RRF_K, Candidate, retrieve
from app.search.rerank import Affinity, build_affinity, relevance_from_rrf, score_job
from app.search.schema import (
    JobOut,
    ProfileSignals,
    RetrievalInfo,
    ScoredJob,
    SearchRequest,
    SearchResponse,
)
from app.services import users as users_svc

POOL_SIZE = 50


class InvalidSearch(ValueError):
    pass


def job_to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id, title=job.title, company=job.company.name, location=job.location, country=job.country,
        work_mode=job.work_mode, job_type=job.job_type, seniority=job.seniority,
        salary_min=job.salary_min, salary_max=job.salary_max, salary_currency=job.salary_currency,
        salary_period=job.salary_period, posted_at=job.posted_at, url=job.url, source=job.source,
        skills=sorted(s.name for s in job.skills), snippet=_snippet(job.description),
    )


def _snippet(text: str, n: int = 280) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1].rstrip() + "…"


def expand_query(query: str) -> tuple[str, list[str]]:
    """Append canonical skill names for aliases used in the query ("k8s" → kubernetes, "large language
    models" → llm), so keyword search matches how postings spell things. Returns (text, added_terms)."""
    low = query.lower()
    added = [s for s in extract_skills(query) if s not in low]
    return (f"{query} {' '.join(added)}" if added else query), added


def profile_query(p: ProfileSignals | None) -> str:
    """Retrieval text for 'recommend me jobs' (no explicit query): who the user wants to be + what they know."""
    if not p:
        return ""
    return " ".join([*p.target_roles, *p.skills[:8]]).strip()


def search_jobs(
    session: Session, embedder: Embedder, req: SearchRequest, *, now: datetime | None = None
) -> SearchResponse:
    t0 = time.perf_counter()
    now = now or datetime.now(timezone.utc)

    profile = req.profile
    affinity: Affinity | None = None
    dismissed_ids: list[int] = []
    saved_ids: set[int] = set()
    if req.user_id:
        if profile is None:
            profile = users_svc.profile_signals(users_svc.get_profile(session, req.user_id))
        saved = users_svc.feedback_jobs(session, req.user_id, "saved")
        dismissed = users_svc.feedback_jobs(session, req.user_id, "dismissed")
        affinity = build_affinity(saved, dismissed)
        dismissed_ids = [j.id for j in dismissed]  # never show what they already said no to
        saved_ids = {j.id for j in saved}

    query = req.query.strip() or profile_query(profile)
    if not query:
        raise InvalidSearch("provide a query, or a profile with skills/target roles to base recommendations on")

    filters = req.filters
    if dismissed_ids:
        filters = filters.model_copy(update={"exclude_job_ids": [*(filters.exclude_job_ids or []), *dismissed_ids]})

    pool = POOL_SIZE if req.rerank else req.limit
    retrieval_text, expansions = expand_query(query) if req.expand else (query, [])
    vector_weight = getattr(embedder, "rrf_weight", 1.0)  # how much to trust this embedder's ranking vs keyword
    cands = retrieve(session, embedder, retrieval_text, filters, mode=req.mode, limit=max(pool, req.limit), now=now,
                     vector_weight=vector_weight)
    jobs = _load_jobs(session, [c.job_id for c in cands])
    cands = [c for c in cands if c.job_id in jobs]

    # Best possible fused score = rank 1 in every active retriever, each at its weight.
    max_weight = {"hybrid": vector_weight + 1.0, "vector": vector_weight, "keyword": 1.0}[req.mode]
    rel = [relevance_from_rrf(c.rrf, max_weight, RRF_K) for c in cands]
    personalised = bool(profile and any((profile.skills, profile.seniority, profile.work_mode, profile.job_type,
                                         profile.preferred_locations, profile.min_salary))) or bool(affinity)
    scored: list[ScoredJob] = []
    for c, r in zip(cands, rel, strict=True):
        job = jobs[c.job_id]
        if req.rerank:
            score, comps, reasons, caveats = score_job(job, r, profile, affinity, now)
        else:
            score, comps, reasons, caveats = r, [], [], []
        scored.append(_scored(job, c, score, r, comps, reasons, caveats, job.id in saved_ids))

    if req.rerank:
        scored.sort(key=lambda s: (-s.score, s.job.id))
    return SearchResponse(
        query_used=query, expansions=expansions, mode=req.mode, reranked=req.rerank, personalised=personalised and req.rerank,
        candidates=len(cands), took_ms=round((time.perf_counter() - t0) * 1000, 1), results=scored[: req.limit],
    )


def _scored(job, c: Candidate, score, rel, comps, reasons, caveats, saved) -> ScoredJob:
    return ScoredJob(
        job=job_to_out(job), score=round(score, 4), relevance=round(rel, 4), components=comps,
        reasons=reasons, caveats=caveats, saved=saved,
        retrieval=RetrievalInfo(vector_rank=c.vector_rank, keyword_rank=c.keyword_rank, rrf=round(c.rrf, 5)),
    )


def _load_jobs(session: Session, ids: list[int]) -> dict[int, Job]:
    if not ids:
        return {}
    rows = session.scalars(select(Job).where(Job.id.in_(ids)).options(selectinload(Job.company)))
    return {j.id: j for j in rows}

