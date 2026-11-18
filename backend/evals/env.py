"""A throwaway, fully-seeded database for evals (embedded Postgres + pgvector unless a URL is given)."""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from alembic import command
from app.db.models import Job
from app.ingest.pipeline import reembed_all, run_ingestion
from app.ingest.sources import SeedSource
from app.search.embedder import Embedder
from evals.corpus import load_corpus

BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass
class EvalEnv:
    session: Session
    records: dict[str, dict]  # external_id -> ground-truth record
    id_to_record: dict[int, dict]  # job id -> record (seed jobs only)
    now: datetime
    embedder: Embedder

    def use_embedder(self, embedder: Embedder) -> None:
        """Switch embedding model/recipe: rebuild every vector with it."""
        if embedder is not self.embedder:
            reembed_all(self.session, embedder)
            self.embedder = embedder


@contextmanager
def eval_env(embedder: Embedder, database_url: str | None = None) -> Iterator[EvalEnv]:
    server = None
    if database_url is None:
        import pgserver  # dev dependency (requirements-dev.txt)

        server = pgserver.get_server(tempfile.mkdtemp(prefix="pathfinder-eval-"))
        database_url = server.get_uri().replace("postgresql://", "postgresql+psycopg://")
    try:
        cfg = Config(str(BACKEND_DIR / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
        cfg.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(cfg, "head")
        engine = create_engine(database_url)
        with Session(engine, expire_on_commit=False) as session:
            run = run_ingestion(session, SeedSource(), embedder, force=True)
            assert run and run.status == "success", run.error if run else "ingestion skipped"
            records = load_corpus()
            ids = dict(session.execute(select(Job.id, Job.external_id).where(Job.source == "seed")).all())
            yield EvalEnv(session, records, {i: records[e] for i, e in ids.items() if e in records},
                          datetime.now(timezone.utc), embedder)
        engine.dispose()
    finally:
        if server is not None:
            server.cleanup()
