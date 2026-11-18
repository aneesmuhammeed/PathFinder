from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import IngestionRun, Job
from app.db.session import get_db

router = APIRouter(tags=["stats"])


@router.get("/stats")
def stats(db: Session = Depends(get_db)) -> dict:
    """Index health: counts per source and the latest ingestion run for each."""
    per_source = db.execute(
        select(
            Job.source,
            func.count().filter(Job.expired_at.is_(None)).label("active"),
            func.count().filter(Job.expired_at.is_not(None)).label("expired"),
            func.count().filter(Job.embedding.is_not(None), Job.expired_at.is_(None)).label("embedded"),
        ).group_by(Job.source)
    ).all()
    latest_ids = select(func.max(IngestionRun.id)).group_by(IngestionRun.source)
    runs = db.scalars(select(IngestionRun).where(IngestionRun.id.in_(latest_ids))).all()
    return {
        "sources": {
            r.source: {"active": r.active, "expired": r.expired, "embedded": r.embedded} for r in per_source
        },
        "last_runs": {
            r.source: {
                "status": r.status, "started_at": r.started_at, "finished_at": r.finished_at,
                "fetched": r.fetched, "inserted": r.inserted, "updated": r.updated,
                "unchanged": r.unchanged, "expired": r.expired, "embedded": r.embedded, "error": r.error,
            }
            for r in runs
        },
    }
