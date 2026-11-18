"""Relevance checks on the full 520-job seed corpus, scored against its known ground-truth role families.

These use the *lexical* hash embedder, so they prove the retrieval/fusion/filter machinery works end to end;
they do not claim semantic understanding. (A real embedding model is evaluated separately in Phase 5.)
"""

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Job
from app.ingest.pipeline import run_ingestion
from app.ingest.sources import SeedSource
from app.search.schema import ProfileSignals, SearchFilters, SearchRequest
from app.search.service import search_jobs
from tests.helpers import EMB, NOW

SEED = Path(__file__).resolve().parents[2] / "data" / "seed_jobs.jsonl"


@pytest.fixture(scope="module")
def seeded(engine):
    """Module-scoped DB session with the seed ingested, rolled back afterwards (invisible to other tests)."""
    conn = engine.connect()
    trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    run_ingestion(session, SeedSource(today=NOW), EMB, now=NOW)
    families = {}
    for line in SEED.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        families[r["external_id"]] = r["role_family"]
    yield session, families
    session.close()
    trans.rollback()
    conn.close()


def precision_at(seeded, query, wanted: set[str], k=10, **kw):
    session, fam = seeded
    res = search_jobs(session, EMB, SearchRequest(query=query, limit=k, **kw), now=NOW)
    ext = dict(session.execute(select(Job.id, Job.external_id).where(Job.id.in_([r.job.id for r in res.results]))).all())
    hits = sum(fam[ext[r.job.id]] in wanted for r in res.results)
    return hits / k


@pytest.mark.parametrize("query,families,minimum", [
    ("LLM agents with tool use and RAG", {"llm"}, 0.8),
    ("react frontend developer", {"frontend", "fullstack"}, 0.8),
    ("kubernetes terraform devops engineer", {"devops"}, 0.7),
    ("android kotlin mobile app developer", {"mobile"}, 0.7),
    ("selenium pytest test automation", {"qa"}, 0.7),
    ("figma ux design research", {"design"}, 0.7),
    ("spark airflow data pipelines", {"data_eng"}, 0.7),
    ("machine learning pytorch model training", {"ml", "llm"}, 0.8),
])
def test_top10_precision_by_role_family(seeded, query, families, minimum):
    assert precision_at(seeded, query, families) >= minimum


@pytest.mark.parametrize("query,families,filters", [
    # Short, role-only queries are the hard case for a hashed embedder (regression: vector retrieval used to
    # return backend jobs for "frontend engineer", and RRF let that noise into the fused ranking).
    ("frontend engineer", {"frontend", "fullstack"}, SearchFilters(work_mode=["remote"])),
    ("backend engineer", {"backend", "fullstack"}, SearchFilters(work_mode=["remote"])),
    ("data scientist", {"ml", "llm", "analyst"}, None),
    ("product manager", {"pm"}, None),
    ("mobile engineer", {"mobile"}, None),
])
def test_short_role_queries_stay_on_topic(seeded, query, families, filters):
    assert precision_at(seeded, query, families, filters=filters or SearchFilters()) >= 0.7


def test_hash_embedder_recipe_weights_the_title_and_caps_the_description():
    from app.search.embedder import HashEmbedder, job_embedding_text

    kw = dict(title="Frontend Engineer", company="C", location="Pune", seniority="mid", job_type="full_time",
              work_mode="remote", skills=["react"], description="x" * 5000)
    text = job_embedding_text(**kw, **HashEmbedder.doc_params)
    assert text.count("Frontend Engineer") == 2 and 2000 < len(text) < 2200


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_every_mode_is_competent_on_clear_queries(seeded, mode):
    assert precision_at(seeded, "react frontend developer", {"frontend", "fullstack"}, mode=mode, rerank=False) >= 0.7


def test_filters_hold_on_full_corpus(seeded):
    session, _ = seeded
    res = search_jobs(session, EMB, SearchRequest(
        query="software engineer", limit=20,
        filters=SearchFilters(job_type=["internship"], locations=["Gurugram"])), now=NOW)
    assert res.results, "expected Gurugram internships in the seed"
    assert all(r.job.job_type == "internship" and "Gurugram" in r.job.location for r in res.results)


def test_intern_profile_surfaces_internships_without_being_told_to(seeded):
    session, _ = seeded
    me = ProfileSignals(skills=["python", "sql", "fastapi"], seniority="intern", job_type="internship")
    res = search_jobs(session, EMB, SearchRequest(query="python backend", profile=me, limit=10), now=NOW)
    plain = search_jobs(session, EMB, SearchRequest(query="python backend", limit=10), now=NOW)
    n = lambda r: sum(x.job.job_type == "internship" for x in r.results)  # noqa: E731
    assert n(res) >= 8 and n(res) > n(plain)


def test_only_active_jobs_and_sane_latency(seeded):
    session, _ = seeded
    res = search_jobs(session, EMB, SearchRequest(query="python", limit=50), now=NOW)
    assert res.candidates == 50 and res.took_ms < 2000
