"""First-stage retrieval: pgvector (semantic) + Postgres full-text (keyword), fused with RRF.

Why two retrievers? Embeddings catch meaning ("LLM agent" ≈ "AI engineer") but blur exact tokens
("C++", "FastAPI", a specific city). Full-text catches exact terms but misses paraphrases. Reciprocal
Rank Fusion merges the two *rankings* without needing their raw scores to be comparable.

Note on filtered ANN: HNSW is an approximate index; with a restrictive WHERE the index may return
fewer than `limit` rows. We raise `hnsw.ef_search` to compensate, and the keyword side still fills in.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import ColumnElement, Text, and_, cast, exists, func, literal_column, or_, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.orm import Session

from app.db.models import Company, Job, JobSkill, Skill
from app.search.embedder import Embedder
from app.search.schema import SearchFilters

RRF_K = 60
EF_SEARCH = 100


@dataclass
class Candidate:
    job_id: int
    rrf: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K, weights: Sequence[float] | None = None) -> dict[int, float]:
    """(Weighted) Reciprocal Rank Fusion: score(d) = Σ w_i / (k + rank_i(d)), ranks starting at 1."""
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings), strict=True):  # one weight per ranking
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return scores


def filter_conditions(f: SearchFilters, now: datetime | None = None) -> list[ColumnElement[bool]]:
    conds: list[ColumnElement[bool]] = [Job.expired_at.is_(None)]
    if f.seniority:
        conds.append(Job.seniority.in_(f.seniority))
    if f.job_type:
        conds.append(Job.job_type.in_(f.job_type))
    if f.work_mode:
        conds.append(Job.work_mode.in_(f.work_mode))
    if f.locations:
        loc = or_(*[
            or_(Job.location.ilike(f"%{_like(x)}%", escape="\\"), Job.country.ilike(f"%{_like(x)}%", escape="\\"))
            for x in f.locations
        ])
        conds.append(or_(loc, Job.work_mode == "remote") if f.include_remote else loc)
    if f.skills_any:
        conds.append(
            exists().where(JobSkill.job_id == Job.id, JobSkill.skill_id == Skill.id, Skill.name.in_(f.skills_any))
        )
    if f.posted_within_days:
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=f.posted_within_days)
        conds.append(Job.posted_at >= cutoff)
    if f.company:
        conds.append(Job.company.has(Company.name.icontains(f.company.strip(), autoescape=True)))
    if f.exclude_job_ids:
        conds.append(Job.id.not_in(f.exclude_job_ids))
    return conds


def _like(s: str) -> str:
    """Escape LIKE wildcards in user input."""
    return s.strip().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def _has_terms(query: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", query))


def vector_ranking(session: Session, embedder: Embedder, query: str, conds, limit: int) -> list[int]:
    qvec = embedder.embed_query(query)
    session.execute(text(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}"))
    dist = Job.embedding.cosine_distance(qvec)
    stmt = select(Job.id).where(and_(*conds), Job.embedding.is_not(None)).order_by(dist, Job.id).limit(limit)
    return list(session.scalars(stmt))


def keyword_ranking(session: Session, query: str, conds, limit: int) -> list[int]:
    """OR-semantics full-text: any query term can match; more/rarer matches rank higher.

    `plainto_tsquery` safely parses arbitrary user text (no query-syntax injection); swapping its AND
    operators for OR means "python backend engineer pune" doesn't require every word to be present.
    """
    if not _has_terms(query):
        return []
    english = literal_column("'english'::regconfig")
    tsq = cast(func.replace(cast(func.plainto_tsquery(english, query), Text), "&", "|"), TSQUERY)
    rank = func.ts_rank_cd(Job.search_tsv, tsq, 1)  # 1 = divide by document length
    stmt = select(Job.id).where(and_(*conds), Job.search_tsv.op("@@")(tsq)).order_by(rank.desc(), Job.id).limit(limit)
    return list(session.scalars(stmt))


def retrieve(
    session: Session,
    embedder: Embedder,
    query: str,
    filters: SearchFilters,
    *,
    mode: str = "hybrid",
    limit: int = 50,
    now: datetime | None = None,
    extra_conditions: Sequence[ColumnElement[bool]] = (),
    vector_weight: float = 1.0,
) -> list[Candidate]:
    conds = [*filter_conditions(filters, now), *extra_conditions]
    vec = vector_ranking(session, embedder, query, conds, limit) if mode in ("hybrid", "vector") else []
    kw = keyword_ranking(session, query, conds, limit) if mode in ("hybrid", "keyword") else []
    lists = [(r, w) for r, w in ((vec, vector_weight), (kw, 1.0)) if r]
    fused = rrf_fuse([r for r, _ in lists], weights=[w for _, w in lists])
    vrank = {j: i for i, j in enumerate(vec, 1)}
    krank = {j: i for i, j in enumerate(kw, 1)}
    ordered = sorted(fused, key=lambda j: (-fused[j], j))[:limit]
    return [Candidate(j, fused[j], vrank.get(j), krank.get(j)) for j in ordered]
