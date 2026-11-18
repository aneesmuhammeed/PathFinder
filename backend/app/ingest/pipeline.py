"""Ingestion pipeline: fetch → dedupe → idempotent upsert → incremental embed → expire → audit.

Guarantees
----------
* Idempotent: running the same input twice changes nothing and embeds nothing (content hash).
* Incremental: only new/changed jobs are (re-)embedded; `embedded_hash` tracks which content a
  vector was built from, so a crash between upsert and embed self-heals on the next run.
* Safe expiry: snapshot sources expire what's missing, but a circuit breaker refuses to expire
  when the fetch looks suspiciously small (API outage / partial response). Rolling-window sources
  expire only after a grace period.
* Auditable: every run writes an `ingestion_runs` row (counts, status, error).
"""

import logging
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.db.models import Company, IngestionRun, Job, Skill
from app.ingest.schema import NormalizedJob
from app.ingest.skills import CATEGORY
from app.ingest.sources.base import JobSource
from app.search.embedder import Embedder, job_embedding_text

log = logging.getLogger("pathfinder.ingest")

CHUNK = 200
BREAKER_MIN_ACTIVE = 20  # below this many active jobs the ratio check is meaningless


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _chunks(items: Sequence, n: int) -> Iterator[Sequence]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def dedupe(jobs: Iterable[NormalizedJob]) -> list[NormalizedJob]:
    """Same external_id twice in one payload: last one wins."""
    by_id: dict[str, NormalizedJob] = {}
    for j in jobs:
        by_id[j.external_id] = j
    return list(by_id.values())


def _apply(job: Job, nj: NormalizedJob, company: Company, skills: list[Skill], content_hash: str) -> None:
    job.company = company
    job.title, job.description, job.url = nj.title, nj.description, nj.url
    job.location, job.country = nj.location, nj.country
    job.work_mode, job.job_type, job.seniority = nj.work_mode, nj.job_type, nj.seniority
    job.salary_min, job.salary_max = nj.salary_min, nj.salary_max
    job.salary_currency, job.salary_period = nj.salary_currency, nj.salary_period
    job.posted_at = nj.posted_at
    job.skills = skills
    job.content_hash = content_hash


def _get_or_create(session: Session, model, names: set[str], extra=None) -> dict:
    found = {o.name: o for o in session.scalars(select(model).where(model.name.in_(names)))}
    for name in names - found.keys():
        obj = model(name=name, **(extra(name) if extra else {}))
        session.add(obj)
        found[name] = obj
    session.flush()
    return found


def _upsert(session: Session, source: str, jobs: list[NormalizedJob], now: datetime) -> dict[str, int]:
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for chunk in _chunks(jobs, CHUNK):
        existing = {
            j.external_id: j
            for j in session.scalars(
                select(Job).where(Job.source == source, Job.external_id.in_([c.external_id for c in chunk]))
            )
        }
        companies = _get_or_create(session, Company, {c.company for c in chunk})
        skills = _get_or_create(
            session, Skill, {s for c in chunk for s in c.skills}, extra=lambda n: {"category": CATEGORY.get(n)}
        )
        touched: list[int] = []
        for nj in chunk:
            h = nj.content_hash()
            row = existing.get(nj.external_id)
            sk = [skills[s] for s in nj.skills]
            if row is None:
                row = Job(source=source, external_id=nj.external_id, first_seen_at=now, last_seen_at=now)
                _apply(row, nj, companies[nj.company], sk, h)
                session.add(row)
                stats["inserted"] += 1
            elif row.content_hash != h or row.expired_at is not None:
                # Changed content, or a previously-expired job that came back.
                _apply(row, nj, companies[nj.company], sk, h)
                row.last_seen_at, row.expired_at = now, None
                stats["updated"] += 1
            else:
                touched.append(row.id)
                stats["unchanged"] += 1
        if touched:  # one bulk UPDATE instead of N row updates
            session.execute(update(Job).where(Job.id.in_(touched)).values(last_seen_at=now))
        session.flush()
    return stats


def reembed_all(session: Session, embedder: Embedder, source: str | None = None, batch_size: int = 64) -> int:
    """Invalidate and rebuild every active vector (use after switching embedding model)."""
    sources = [source] if source else list(session.scalars(select(Job.source).distinct()))
    session.execute(
        update(Job).where(Job.expired_at.is_(None), *([Job.source == source] if source else []))
        .values(embedded_hash=None)
    )
    session.commit()
    return sum(_embed_pending(session, s, embedder, batch_size) for s in sources)


def _embed_pending(session: Session, source: str, embedder: Embedder, batch_size: int) -> int:
    """Embed active jobs whose vector is missing or built from older content."""
    total = 0
    while True:
        batch = session.scalars(
            select(Job)
            .where(
                Job.source == source,
                Job.expired_at.is_(None),
                or_(Job.embedded_hash.is_(None), Job.embedded_hash != Job.content_hash),
            )
            .options(selectinload(Job.company))
            .order_by(Job.id)
            .limit(batch_size)
        ).all()
        if not batch:
            return total
        texts = [
            job_embedding_text(
                title=j.title, company=j.company.name, location=j.location, seniority=j.seniority,
                job_type=j.job_type, work_mode=j.work_mode, skills=[s.name for s in j.skills],
                description=j.description, **getattr(embedder, "doc_params", {}),
            )
            for j in batch
        ]
        for j, vec in zip(batch, embedder.embed(texts), strict=True):
            j.embedding, j.embedded_hash = vec, j.content_hash
        session.commit()
        total += len(batch)


def _expire(session: Session, source: JobSource, fetched: int, now: datetime, stale_days: int,
            min_fetch_ratio: float) -> tuple[int, str | None]:
    active = session.scalar(
        select(func.count()).select_from(Job).where(Job.source == source.name, Job.expired_at.is_(None))
    ) or 0
    if source.snapshot:
        if active >= BREAKER_MIN_ACTIVE and fetched < min_fetch_ratio * active:
            msg = f"expiry skipped: fetched {fetched} but {active} active (< {min_fetch_ratio:.0%}); suspected partial fetch"
            log.warning(msg, extra={"source": source.name})
            return 0, msg
        cutoff = now
    else:
        cutoff = now - timedelta(days=stale_days)
    result = session.execute(
        update(Job)
        .where(Job.source == source.name, Job.expired_at.is_(None), Job.last_seen_at < cutoff)
        .values(expired_at=now)
    )
    return result.rowcount or 0, None


def run_ingestion(
    session: Session,
    source: JobSource,
    embedder: Embedder,
    *,
    now: datetime | None = None,
    expire: bool = True,
    force: bool = False,
    min_fetch_ratio: float = 0.5,
    batch_size: int = 64,
) -> IngestionRun | None:
    """Run one ingestion cycle for `source`. Returns the audit row, or None if skipped by rate limit.

    Source failures never raise: the run is recorded as failed and nothing is expired.
    """
    settings = get_settings()
    now = now or _utcnow()

    if not force and source.min_interval:
        last = session.scalar(
            select(func.max(IngestionRun.started_at)).where(
                IngestionRun.source == source.name, IngestionRun.status == "success"
            )
        )
        if last and now - last < source.min_interval:
            log.info("skipped: ran recently", extra={"source": source.name, "last": last.isoformat()})
            return None

    run = IngestionRun(source=source.name, started_at=now, status="running")
    session.add(run)
    session.commit()
    run_id = run.id

    try:
        jobs = dedupe(source.fetch())
        run.fetched = len(jobs)
        stats = _upsert(session, source.name, jobs, now)
        session.commit()
        run.inserted, run.updated, run.unchanged = stats["inserted"], stats["updated"], stats["unchanged"]
        run.embedded = _embed_pending(session, source.name, embedder, batch_size)
        if expire:
            run.expired, note = _expire(session, source, len(jobs), now, settings.job_stale_after_days, min_fetch_ratio)
            run.error = note
        run.status = "success"
    except Exception as exc:  # noqa: BLE001 - a bad source must not take the process down
        log.exception("ingestion failed", extra={"source": source.name})
        session.rollback()
        run = session.get(IngestionRun, run_id)
        run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
    run.finished_at = _utcnow()
    session.commit()
    log.info(
        "ingestion finished",
        extra={k: getattr(run, k) for k in ("source", "status", "fetched", "inserted", "updated", "unchanged", "expired", "embedded")},
    )
    return run
