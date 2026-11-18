from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from app.db.models import Company, IngestionRun, Job, Skill
from app.ingest.pipeline import dedupe, run_ingestion
from app.ingest.schema import NormalizedJob
from app.ingest.sources import SeedSource
from app.search.embedder import HashEmbedder

T0 = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def at(hours=0, days=0):
    return T0 + timedelta(hours=hours, days=days)


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__(384)
        self.texts_embedded = 0

    def embed(self, texts):
        self.texts_embedded += len(texts)
        return super().embed(texts)


class FakeSource:
    def __init__(self, jobs=None, *, snapshot=True, min_interval=None, error=None):
        self.name, self.snapshot, self.min_interval = "fake", snapshot, min_interval
        self.jobs, self.error = jobs or [], error

    def fetch(self):
        if self.error:
            raise self.error
        return list(self.jobs)


def mk(i, **kw):
    base = dict(source="fake", external_id=f"j{i}", company=f"Co{i % 3}", title=f"Backend Engineer {i}",
                description=f"Build python services number {i} with postgres.", skills=["python", "postgres"])
    base.update(kw)
    return NormalizedJob(**base)


def count(db, *where):
    return db.scalar(select(func.count()).select_from(Job).where(Job.source == "fake", *where))


def run(db, src, emb=None, **kw):
    return run_ingestion(db, src, emb or CountingEmbedder(), **kw)


def test_first_run_inserts_and_embeds(db):
    emb = CountingEmbedder()
    r = run(db, FakeSource([mk(i) for i in range(10)]), emb, now=at())
    assert (r.status, r.fetched, r.inserted, r.updated, r.unchanged, r.embedded, r.expired) == ("success", 10, 10, 0, 0, 10, 0)
    assert emb.texts_embedded == 10
    j = db.scalars(select(Job).where(Job.source == "fake")).first()
    assert len(j.embedding) == 384 and j.embedded_hash == j.content_hash and r.finished_at


def test_rerun_is_idempotent_and_embeds_nothing(db):
    src, emb = FakeSource([mk(i) for i in range(10)]), CountingEmbedder()
    run(db, src, emb, now=at())
    emb.texts_embedded = 0
    r = run(db, src, emb, now=at(hours=1))
    assert (r.inserted, r.updated, r.unchanged, r.embedded, r.expired) == (0, 0, 10, 0, 0)
    assert emb.texts_embedded == 0
    assert count(db) == 10


def test_only_changed_job_is_updated_and_reembedded(db):
    emb = CountingEmbedder()
    jobs = [mk(i) for i in range(5)]
    run(db, FakeSource(jobs), emb, now=at())
    emb.texts_embedded = 0
    jobs[2] = mk(2, description="Completely rewritten description about kubernetes platforms.")
    r = run(db, FakeSource(jobs), emb, now=at(hours=1))
    assert (r.inserted, r.updated, r.unchanged, r.embedded) == (0, 1, 4, 1)
    assert emb.texts_embedded == 1
    row = db.scalar(select(Job).where(Job.external_id == "j2", Job.source == "fake"))
    assert "kubernetes" in row.description and row.embedded_hash == row.content_hash


def test_posted_at_change_alone_is_not_an_update(db):
    run(db, FakeSource([mk(1, posted_at=at())]), now=at())
    r = run(db, FakeSource([mk(1, posted_at=at(days=3))]), now=at(hours=1))
    assert (r.updated, r.unchanged, r.embedded) == (0, 1, 0)


def test_missing_jobs_expire_and_reappearing_jobs_reactivate(db):
    jobs = [mk(i) for i in range(5)]
    run(db, FakeSource(jobs), now=at())
    r = run(db, FakeSource(jobs[:4]), now=at(hours=1))
    assert r.expired == 1 and count(db, Job.expired_at.is_not(None)) == 1 and count(db, Job.expired_at.is_(None)) == 4
    r = run(db, FakeSource(jobs), now=at(hours=2))
    assert (r.updated, r.expired, r.inserted) == (1, 0, 0)  # same row revived, not duplicated
    assert count(db) == 5 and count(db, Job.expired_at.is_not(None)) == 0


def test_circuit_breaker_blocks_mass_expiry_on_partial_fetch(db):
    jobs = [mk(i) for i in range(40)]
    run(db, FakeSource(jobs), now=at())
    r = run(db, FakeSource(jobs[:5]), now=at(hours=1))  # API returned 5 of 40: smells like an outage
    assert r.status == "success" and r.expired == 0 and "expiry skipped" in r.error
    assert count(db, Job.expired_at.is_(None)) == 40


def test_rolling_window_source_expires_only_after_grace_period(db):
    jobs = [mk(i) for i in range(5)]
    run(db, FakeSource(jobs, snapshot=False), now=at())
    r = run(db, FakeSource(jobs[:3], snapshot=False), now=at(days=3))  # 2 scrolled out of window
    assert r.expired == 0
    r = run(db, FakeSource(jobs[:3], snapshot=False), now=at(days=20))
    assert r.expired == 2  # not seen for >14 days


def test_source_failure_is_recorded_not_raised_and_expires_nothing(db):
    jobs = [mk(i) for i in range(5)]
    run(db, FakeSource(jobs), now=at())
    r = run(db, FakeSource(error=RuntimeError("API down")), now=at(hours=1))
    assert r.status == "failed" and "API down" in r.error and r.finished_at
    assert count(db, Job.expired_at.is_(None)) == 5


def test_duplicate_ids_in_payload_last_wins(db):
    r = run(db, FakeSource([mk(1, title="Old"), mk(1, title="New")]), now=at())
    assert r.fetched == 1 and db.scalar(select(Job.title).where(Job.external_id == "j1")) == "New"
    assert len(dedupe([mk(1), mk(1), mk(2)])) == 2


def test_companies_and_skills_are_shared_not_duplicated(db):
    run(db, FakeSource([mk(i) for i in range(9)]), now=at())
    assert db.scalar(select(func.count()).select_from(Company).where(Company.name.in_(["Co0", "Co1", "Co2"]))) == 3
    assert db.scalar(select(func.count()).select_from(Skill).where(Skill.name == "postgresql")) == 1
    j = db.scalar(select(Job).where(Job.external_id == "j0", Job.source == "fake"))
    assert {s.name for s in j.skills} == {"python", "postgresql"}


def test_rate_limit_skips_without_creating_a_run(db):
    src = FakeSource([mk(1)], min_interval=timedelta(hours=6))
    assert run(db, src, now=at()) is not None
    n = db.scalar(select(func.count()).select_from(IngestionRun).where(IngestionRun.source == "fake"))
    assert run(db, src, now=at(hours=1)) is None
    assert run(db, src, now=at(hours=1), force=True) is not None
    assert run(db, src, now=at(hours=7)) is not None
    assert db.scalar(select(func.count()).select_from(IngestionRun).where(IngestionRun.source == "fake")) == n + 2


def test_unembedded_rows_self_heal_on_next_run(db):
    jobs = [mk(i) for i in range(3)]
    run(db, FakeSource(jobs), now=at())
    db.execute(Job.__table__.update().where(Job.source == "fake").values(embedding=None, embedded_hash=None))
    r = run(db, FakeSource(jobs), now=at(hours=1))
    assert r.unchanged == 3 and r.embedded == 3


def test_reembed_all_rebuilds_vectors_with_new_embedder(db):
    from app.ingest.pipeline import reembed_all

    run(db, FakeSource([mk(i) for i in range(4)]), now=at())
    before = db.scalar(select(Job.embedding).where(Job.external_id == "j0", Job.source == "fake"))

    class Other(HashEmbedder):  # stands in for switching to a different model
        def embed(self, texts):
            return [[1.0] + [0.0] * 383 for _ in texts]

    n = reembed_all(db, Other(384), source="fake")
    after = db.scalar(select(Job.embedding).where(Job.external_id == "j0", Job.source == "fake"))
    assert n == 4 and list(after) != list(before) and after[0] == 1.0


def test_full_seed_corpus_ingests_and_is_idempotent(db):
    emb = CountingEmbedder()
    r = run_ingestion(db, SeedSource(), emb, now=at())
    assert r.status == "success" and r.inserted >= 500 and r.embedded == r.inserted
    assert db.scalar(select(func.count()).select_from(Job).where(Job.source == "seed", Job.embedding.is_(None))) == 0
    r2 = run_ingestion(db, SeedSource(), emb, now=at(hours=1))
    assert (r2.inserted, r2.updated, r2.embedded, r2.expired) == (0, 0, 0, 0) and r2.unchanged == r.inserted
    assert db.scalar(select(func.count()).select_from(Skill)) > 50


def test_stats_endpoint_reports_sources_and_last_run(db, client, engine):
    # The API uses its own connection, so commit real data and clean up afterwards.
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        run_ingestion(s, FakeSource([mk(i) for i in range(3)]), CountingEmbedder(), now=at())
    try:
        body = client.get("/stats").json()
        assert body["sources"]["fake"] == {"active": 3, "expired": 0, "embedded": 3}
        assert body["last_runs"]["fake"]["status"] == "success"
    finally:
        with Session(engine) as s:
            s.execute(Job.__table__.delete().where(Job.source == "fake"))  # job_skills rows cascade
            s.execute(IngestionRun.__table__.delete().where(IngestionRun.source == "fake"))
            s.execute(Skill.__table__.delete().where(Skill.name.in_(["python", "postgresql"])))
            s.execute(Company.__table__.delete().where(Company.name.in_(["Co0", "Co1", "Co2"])))
            s.commit()
