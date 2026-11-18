"""Test fixtures: a real PostgreSQL 16 + pgvector (embedded via `pgserver`, no Docker needed).

If TEST_DATABASE_URL is set (e.g. in CI against the pgvector service container) it is used instead.
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from alembic import command

BACKEND_DIR = Path(__file__).resolve().parent.parent

# pgserver logs at interpreter shutdown, after pytest has closed captured stdout.
logging.raiseExceptions = False


@pytest.fixture(scope="session")
def database_url():
    external = os.environ.get("TEST_DATABASE_URL")
    if external:
        yield external
        return
    import pgserver

    data_dir = tempfile.mkdtemp(prefix="pathfinder-pg-")
    server = pgserver.get_server(data_dir)
    yield server.get_uri().replace("postgresql://", "postgresql+psycopg://")
    server.cleanup()


@pytest.fixture(scope="session")
def migrated_url(database_url):
    os.environ["DATABASE_URL"] = database_url
    from app.core.config import get_settings
    from app.db.session import reset_engine_cache

    get_settings.cache_clear()
    reset_engine_cache()
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")
    return database_url


@pytest.fixture(scope="session")
def engine(migrated_url):
    eng = create_engine(migrated_url)
    yield eng
    eng.dispose()


@pytest.fixture()
def db(engine):
    """A session whose changes are rolled back after each test."""
    conn = engine.connect()
    trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    yield session
    session.close()
    trans.rollback()
    conn.close()


@pytest.fixture()
def client(migrated_url):
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


def table_names(engine) -> set[str]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")).all()
    return {r[0] for r in rows}
