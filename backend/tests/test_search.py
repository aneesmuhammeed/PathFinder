import pytest
from sqlalchemy import update

from app.db.models import Job
from app.search.hybrid import retrieve, rrf_fuse
from app.search.schema import ProfileSignals, SearchFilters, SearchRequest
from app.search.service import InvalidSearch, expand_query, search_jobs
from app.services import users as svc
from tests.helpers import EMB, NOW, load, nj


def search(db, **kw):
    return search_jobs(db, EMB, SearchRequest(**kw), now=NOW)


def titles(res):
    return [r.job.title for r in res.results]


# ---------------- RRF ----------------

def test_rrf_rewards_agreement_between_rankers():
    s = rrf_fuse([[1, 2], [3, 2]])
    assert max(s, key=s.get) == 2  # second in both beats first in only one
    assert s[2] == pytest.approx(2 / 62)


def test_rrf_is_rank_based_not_score_based():
    assert rrf_fuse([[7, 8, 9]]) == {7: 1 / 61, 8: 1 / 62, 9: 1 / 63}


# ---------------- filters ----------------

@pytest.fixture()
def corpus(db):
    return load(db, [
        nj("sr", "Senior Backend Engineer", ["python", "postgres"], seniority="senior", company="Acme", days_ago=2),
        nj("jr", "Junior Backend Engineer", ["python", "fastapi"], seniority="junior", location="Pune",
           work_mode="hybrid", company="Beta", days_ago=30),
        nj("in", "Backend Engineer Intern", ["python"], seniority="intern", job_type="internship",
           location="Remote (India)", work_mode="remote", days_ago=5),
        nj("fe", "Frontend Engineer", ["react", "css"], seniority="mid", company="Gamma", days_ago=1),
    ])


def ext(res, ids):
    inv = {v: k for k, v in ids.items()}
    return {inv[r.job.id] for r in res.results}


def test_filters(db, corpus):
    q = "backend python services"
    f = lambda **kw: ext(search(db, query=q, filters=SearchFilters(**kw), limit=20), corpus)  # noqa: E731
    assert f(seniority=["senior"]) == {"sr"}
    assert f(job_type=["internship"]) == {"in"}
    assert f(work_mode=["hybrid", "remote"]) == {"jr", "in"}
    assert f(locations=["pune"]) == {"jr"}  # case-insensitive
    assert f(locations=["pune"], include_remote=True) == {"jr", "in"}
    assert f(skills_any=["Postgres"]) == {"sr"}  # alias → canonical postgresql
    assert f(posted_within_days=7) >= {"sr", "in"} and "jr" not in f(posted_within_days=7)
    assert f(company="acm") == {"sr"}
    assert "sr" not in f(exclude_job_ids=[corpus["sr"]])


def test_filter_input_is_not_a_sql_or_like_injection(db, corpus):
    assert search(db, query="backend", filters=SearchFilters(locations=["%"]), limit=20).results == []
    assert search(db, query="backend", filters=SearchFilters(company="'; DROP TABLE jobs;--")).results == []


def test_expired_jobs_never_returned(db, corpus):
    db.execute(update(Job).where(Job.id == corpus["sr"]).values(expired_at=NOW))
    assert "sr" not in ext(search(db, query="senior backend engineer", limit=20), corpus)


# ---------------- retrieval behaviour ----------------

def test_keyword_side_uses_or_semantics(db):
    ids = load(db, [nj("k", "Platform Engineer", ["kubernetes"], desc="Operate kubernetes operators.")])
    res = search(db, query="kubernetes golang cobol fortran", mode="keyword", rerank=False)
    assert [r.job.id for r in res.results] == [ids["k"]]  # matched 1 of 4 terms


def test_weird_query_text_is_safe(db, corpus):
    for q in ["c++ & | ! ( ) :*", "O'Reilly", "'; DROP TABLE jobs;--", "!!!", "a" * 5000]:
        search(db, query=q)  # must not raise
    assert db.query(Job).count() >= 4


def test_job_without_embedding_is_still_found_by_keyword(db):
    ids = load(db, [nj("n", "Quantum Plumber", ["python"], desc="Fix quantum pipes.")], embed=False)
    hy = search(db, query="quantum plumber", mode="hybrid")
    assert hy.results[0].job.id == ids["n"] and hy.results[0].retrieval.vector_rank is None
    assert search(db, query="quantum plumber", mode="vector").results == []


def test_modes_report_which_retriever_found_what(db, corpus):
    r = search(db, query="backend python", mode="hybrid").results[0].retrieval
    assert r.vector_rank and r.keyword_rank and r.rrf > 1 / 61
    assert all(x.retrieval.keyword_rank is None for x in search(db, query="backend python", mode="vector").results)
    assert all(x.retrieval.vector_rank is None for x in search(db, query="backend python", mode="keyword").results)


def test_query_expansion_maps_aliases_to_canonical_terms():
    assert expand_query("k8s and postgres")[1] == ["kubernetes", "postgresql"]
    assert expand_query("kubernetes")[1] == []
    text, added = expand_query("large language models chatbot")
    assert added == ["llm"] and text.endswith("llm")


def test_expansion_helps_keyword_match(db):
    ids = load(db, [nj("l", "AI Engineer", ["llm"], desc="Work on llm systems.")])
    res = search(db, query="large language models", mode="keyword", rerank=False)
    assert [r.job.id for r in res.results] == [ids["l"]] and res.expansions == ["llm"]


# ---------------- personalisation ----------------

@pytest.fixture()
def twins(db):
    # Same title, near-identical text. Only skills / seniority differ.
    return load(db, [
        nj("py", "Software Engineer", ["python", "sql"], seniority="junior", desc="Software engineer building products."),
        nj("jv", "Software Engineer", ["java", "spring boot"], seniority="senior", desc="Software engineer building products."),
    ])


def test_profile_reorders_equally_relevant_jobs(db, twins):
    prof = ProfileSignals(skills=["python", "sql"], seniority="junior")
    res = search(db, query="software engineer", profile=prof)
    assert res.personalised and res.results[0].job.id == twins["py"]
    top = res.results[0]
    assert any("2 of 2" in r for r in top.reasons)
    assert any(c.feature == "skills" and c.value == 1.0 for c in top.components)
    assert any("senior" in c for c in res.results[1].caveats)  # explains why it ranked lower
    flipped = search(db, query="software engineer", profile=ProfileSignals(skills=["java"], seniority="senior"))
    assert flipped.results[0].job.id == twins["jv"]


def test_rerank_off_returns_pure_retrieval_order(db, twins):
    res = search(db, query="software engineer", rerank=False, profile=ProfileSignals(skills=["java"]))
    assert not res.reranked and not res.personalised
    assert all(r.components == [] for r in res.results)


def test_scores_are_bounded_and_components_add_up(db, corpus):
    prof = ProfileSignals(skills=["python"], seniority="junior", work_mode="remote", preferred_locations=["Pune"])
    for r in search(db, query="backend", profile=prof, limit=10).results:
        assert 0 <= r.score <= 1
        assert sum(c.contribution for c in r.components) == pytest.approx(r.score, abs=0.01)
        assert sum(c.weight for c in r.components) == pytest.approx(1.0, abs=0.01)


def test_recommendations_from_profile_alone(db, corpus):
    prof = ProfileSignals(skills=["react", "css"], target_roles=["frontend engineer"])
    res = search(db, query="", profile=prof)
    assert res.query_used.startswith("frontend engineer") and res.results[0].job.id == corpus["fe"]


def test_no_query_and_no_profile_is_rejected(db):
    with pytest.raises(InvalidSearch):
        search(db, query="  ")
    with pytest.raises(InvalidSearch):
        search(db, query="", profile=ProfileSignals())


# ---------------- feedback loop ----------------

def test_dismissed_hidden_saved_flagged_and_affinity_shifts_ranking(db, twins):
    u = svc.create_user(db)
    svc.record_feedback(db, u.id, twins["jv"], "dismissed")
    res = search(db, query="software engineer", user_id=u.id)
    assert [r.job.id for r in res.results] == [twins["py"]]  # dismissed job is gone

    svc.record_feedback(db, u.id, twins["py"], "saved")
    top = search(db, query="software engineer", user_id=u.id).results[0]
    assert top.saved and any(c.feature == "feedback" for c in top.components)


def test_saving_a_dismissed_job_cancels_the_dismissal(db, twins):
    u = svc.create_user(db)
    svc.record_feedback(db, u.id, twins["jv"], "dismissed")
    svc.record_feedback(db, u.id, twins["jv"], "saved")
    svc.record_feedback(db, u.id, twins["jv"], "saved")  # idempotent
    assert [j.id for j in svc.feedback_jobs(db, u.id, "saved")] == [twins["jv"]]
    assert svc.feedback_jobs(db, u.id, "dismissed") == []


def test_stored_profile_is_used_when_user_id_given(db, twins):
    u = svc.create_user(db)
    svc.update_profile(db, u.id, {"skills": ["Java", "Spring Boot"], "seniority": "senior"})
    res = search(db, query="software engineer", user_id=u.id)
    assert res.personalised and res.results[0].job.id == twins["jv"]


def test_inline_profile_overrides_stored(db, twins):
    u = svc.create_user(db)
    svc.update_profile(db, u.id, {"skills": ["java"]})
    res = search(db, query="software engineer", user_id=u.id, profile=ProfileSignals(skills=["python", "sql"]))
    assert res.results[0].job.id == twins["py"]


def test_retrieve_respects_limit_and_returns_ranks(db, corpus):
    c = retrieve(db, EMB, "backend python", SearchFilters(), limit=2, now=NOW)
    assert len(c) == 2 and c[0].rrf >= c[1].rrf
