from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.core.config import get_settings
from app.db.session import get_db
from app.llm.factory import resolve_provider

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness: the process is up. Never touches the DB. `llm` lets a UI say honestly whether it is
    talking to a real model or the keyless demo assistant."""
    return {"status": "ok", "version": __version__, "llm": resolve_provider(get_settings())}


@router.get("/ready")
def ready(response: Response, db: Session = Depends(get_db)) -> dict:
    """Readiness: DB reachable, pgvector installed, schema migrated."""
    checks: dict[str, bool] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
        checks["pgvector"] = bool(
            db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
        )
        checks["migrated"] = bool(db.execute(text("SELECT to_regclass('public.jobs')")).scalar())
    except Exception:  # noqa: BLE001 - readiness must never raise
        checks.setdefault("database", False)
    ok = all(checks.get(k) for k in ("database", "pgvector", "migrated"))
    if not ok:
        response.status_code = 503
    return {"status": "ready" if ok else "not_ready", "checks": checks}
