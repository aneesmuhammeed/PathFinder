"""One command, no Docker, no config: embedded Postgres + migrations + seed data + the API.

    python scripts/dev_stack.py                  # http://127.0.0.1:8000  (data kept in backend/.pgdata)
    python scripts/dev_stack.py --ephemeral      # throwaway database
    python scripts/dev_stack.py --port 8001

Uses the keyless mock assistant unless ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY is set (see .env).
Ctrl+C stops the API and the database cleanly (your data is kept unless --ephemeral).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)  # alembic.ini and .env are resolved relative to backend/


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ephemeral", action="store_true", help="use a temporary database that is deleted on exit")
    p.add_argument("--no-seed", action="store_true", help="don't load the demo jobs if the database is empty")
    a = p.parse_args()

    import pgserver

    data_dir = tempfile.mkdtemp(prefix="pathfinder-dev-") if a.ephemeral else str(BACKEND / ".pgdata")
    server = pgserver.get_server(data_dir, cleanup_mode="delete" if a.ephemeral else "stop")
    os.environ["DATABASE_URL"] = server.get_uri().replace("postgresql://", "postgresql+psycopg://")

    # Imported only now: settings read DATABASE_URL at import time.
    import uvicorn
    from alembic.config import Config
    from sqlalchemy import func, select

    from alembic import command
    from app.db.models import Job
    from app.db.session import get_sessionmaker
    from app.ingest.pipeline import run_ingestion
    from app.ingest.sources import SeedSource
    from app.search.embedder import get_embedder

    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")

    with get_sessionmaker()() as session:
        if not a.no_seed and not session.scalar(select(func.count()).select_from(Job)):
            print("Empty database: loading the 520 demo jobs ...", flush=True)
            run = run_ingestion(session, SeedSource(), get_embedder(), force=True)
            print(f"  {run.status}: {run.inserted} jobs", flush=True)

    from app.core.config import get_settings
    from app.llm.factory import resolve_provider

    print(f"\nPathfinder API on http://{a.host}:{a.port}  (docs: /docs)  assistant: {resolve_provider(get_settings())}\n", flush=True)
    try:
        uvicorn.run("app.main:app", host=a.host, port=a.port, log_level="warning")
    finally:
        server.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
