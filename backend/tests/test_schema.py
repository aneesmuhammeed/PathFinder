import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.db.models import Company, Job, JobFeedback, Skill, User
from tests.conftest import table_names

EXPECTED_TABLES = {
    "companies", "skills", "jobs", "job_skills", "users", "user_profiles",
    "conversations", "messages", "job_feedback", "ingestion_runs", "alembic_version",
}


def make_job(company, **kw):
    base = dict(
        source="test", external_id="1", company=company, title="Backend Engineer",
        description="Build Python APIs with FastAPI and PostgreSQL.", content_hash="h1",
    )
    base.update(kw)
    return Job(**base)


def test_migration_creates_all_tables(engine):
    assert EXPECTED_TABLES <= table_names(engine)


def test_pgvector_and_indexes_present(engine):
    with engine.connect() as c:
        assert c.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar() == 1
        idx = {r[0]: r[1] for r in c.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename='jobs'"))}
    assert "hnsw" in idx["ix_jobs_embedding_hnsw"] and "vector_cosine_ops" in idx["ix_jobs_embedding_hnsw"]
    assert "gin" in idx["ix_jobs_search_tsv"]


def test_job_roundtrip_with_embedding_and_skills(db):
    co = Company(name="Acme")
    job = make_job(co, embedding=[0.1] * 384)
    job.skills = [Skill(name="python"), Skill(name="fastapi")]
    db.add(job)
    db.flush()
    db.refresh(job)
    assert len(job.embedding) == 384
    assert {s.name for s in job.skills} == {"python", "fastapi"}


def test_tsvector_is_generated_and_title_is_searchable(db):
    db.add(make_job(Company(name="Acme"), title="Machine Learning Engineer"))
    db.flush()
    hit = db.execute(
        text("SELECT count(*) FROM jobs WHERE search_tsv @@ websearch_to_tsquery('english', 'machine learning')")
    ).scalar()
    assert hit == 1


def test_vector_cosine_nearest_neighbour(db):
    co = Company(name="Acme")
    near = make_job(co, external_id="near", embedding=[1.0] + [0.0] * 383)
    far = make_job(co, external_id="far", embedding=[0.0] * 383 + [1.0])
    db.add_all([near, far])
    db.flush()
    q = [1.0] + [0.0] * 383
    top = db.execute(select(Job).order_by(Job.embedding.cosine_distance(q)).limit(1)).scalar_one()
    assert top.external_id == "near"


def test_source_external_id_is_unique(db):
    co = Company(name="Acme")
    db.add(make_job(co))
    db.flush()
    db.add(make_job(co))
    with pytest.raises(IntegrityError):
        db.flush()


def test_check_constraints_reject_bad_enums(db):
    db.add(make_job(Company(name="Acme"), seniority="wizard"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_feedback_is_unique_per_user_job_action(db):
    co = Company(name="Acme")
    job, user = make_job(co), User()
    db.add_all([job, user])
    db.flush()
    db.add(JobFeedback(user_id=user.id, job_id=job.id, action="saved"))
    db.flush()
    db.add(JobFeedback(user_id=user.id, job_id=job.id, action="saved"))
    with pytest.raises(IntegrityError):
        db.flush()
