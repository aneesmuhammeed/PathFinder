"""Pure unit tests of the feature scorers: no database."""

from datetime import timedelta

import pytest

from app.db.models import Company, Job, Skill
from app.search.rerank import WEIGHTS, Affinity, build_affinity, relevance_from_rrf, score_job
from app.search.schema import ProfileSignals
from tests.helpers import NOW


def job(skills=("python", "sql"), **kw):
    base = dict(title="T", description="d", seniority="junior", job_type="full_time", work_mode="onsite",
                location="Pune", country="India", company=Company(name="Acme"), posted_at=NOW,
                skills=[Skill(name=s) for s in skills])
    base.update(kw)
    return Job(**base)


def comp(j, profile=None, affinity=None, rel=1.0):
    score, comps, reasons, caveats = score_job(j, rel, profile, affinity, NOW)
    return score, {c.feature: c for c in comps}, reasons, caveats


def test_weights_sum_to_one_and_no_profile_means_only_relevance_and_recency():
    assert all(w > 0 for w in WEIGHTS.values()) and WEIGHTS["relevance"] == max(WEIGHTS.values())  # relevance dominates
    score, c, *_ = comp(job(), None)
    assert set(c) == {"relevance", "recency"}
    assert sum(x.weight for x in c.values()) == pytest.approx(1.0)


def test_sparse_profile_is_not_penalised():
    full = comp(job(), ProfileSignals(skills=["python", "sql"], seniority="junior"))[0]
    sparse = comp(job(), ProfileSignals(skills=["python", "sql"]))[0]
    assert full == pytest.approx(sparse, abs=0.02)  # missing features are dropped, not scored 0


def test_skill_coverage_and_alias_canonicalisation():
    p = ProfileSignals(skills=["Postgres", "Python"])
    _, c, reasons, _ = comp(job(skills=("python", "postgresql", "kafka", "redis")), p)
    assert c["skills"].value == pytest.approx(2 / 4)
    assert "python" in c["skills"].detail and "postgresql" in c["skills"].detail


def test_skill_coverage_denominator_is_capped_for_keyword_stuffed_jobs():
    many = tuple(f"s{i}" for i in range(20))
    _, c, *_ = comp(job(skills=("python", "sql", "react", "go", "java", "rust", "ruby", "php", *many)),
                    ProfileSignals(skills=["python", "sql", "react", "go", "java", "rust", "ruby", "php"]))
    assert c["skills"].value == 1.0  # 8 matches is plenty for a 28-skill posting


@pytest.mark.parametrize("job_level,expected", [("junior", 1.0), ("mid", 0.5), ("intern", 0.5), ("senior", 0.0), ("lead", 0.0)])
def test_seniority_fit(job_level, expected):
    _, c, *_ = comp(job(seniority=job_level), ProfileSignals(seniority="junior"))
    assert c["seniority"].value == expected


def test_work_mode_and_job_type_and_location():
    p = ProfileSignals(work_mode="remote", job_type="internship", preferred_locations=["Gurugram"])
    _, c, _, caveats = comp(job(work_mode="onsite", job_type="full_time", location="Pune"), p)
    assert (c["work_mode"].value, c["job_type"].value, c["location"].value) == (0.0, 0.0, 0.0)
    assert len(caveats) == 3
    _, c, *_ = comp(job(work_mode="hybrid"), ProfileSignals(work_mode="remote"))
    assert c["work_mode"].value == 0.5
    _, c, *_ = comp(job(work_mode="remote", location="Remote (India)"), ProfileSignals(preferred_locations=["Gurugram"]))
    assert c["location"].value == 0.6  # remote isn't blocked by location
    _, c, *_ = comp(job(location="Gurugram"), ProfileSignals(preferred_locations=["gurugram"]))
    assert c["location"].value == 1.0


def test_salary_only_compared_in_inr_and_annualised():
    p = ProfileSignals(min_salary=600_000)
    _, c, *_ = comp(job(salary_max=60_000, salary_currency="INR", salary_period="month"), p)
    assert c["salary"].value == 1.0  # 60k/month = 720k/year
    _, c, *_ = comp(job(salary_max=300_000, salary_currency="INR", salary_period="year"), p)
    assert c["salary"].value == pytest.approx(0.25)
    _, c, *_ = comp(job(salary_max=999_999, salary_currency="USD", salary_period="year"), p)
    assert "salary" not in c  # never compare across currencies
    _, c, *_ = comp(job(), p)
    assert "salary" not in c  # undisclosed → neutral


def test_recency_decays_with_half_life():
    fresh = comp(job(posted_at=NOW))[1]["recency"].value
    three_weeks = comp(job(posted_at=NOW - timedelta(days=21)))[1]["recency"].value
    unknown = comp(job(posted_at=None))[1]["recency"].value
    assert fresh == pytest.approx(1.0) and three_weeks == pytest.approx(0.5) and unknown == 0.3


def test_feedback_affinity_direction():
    liked, disliked = job(skills=("python", "django")), job(skills=("java", "spring boot"))
    aff = build_affinity([liked, liked], [disliked])
    assert comp(job(skills=("python", "django")), None, aff)[1]["feedback"].value > 0.6
    assert comp(job(skills=("java", "spring boot")), None, aff)[1]["feedback"].value < 0.4
    assert "feedback" not in comp(job(), None, Affinity())[1]  # no history → no signal


def test_relevance_is_absolute_not_relative_to_the_pool():
    # first in both retrievers = 1.0; first in only one of two = 0.5
    assert relevance_from_rrf(2 / 61, 2) == pytest.approx(1.0)
    assert relevance_from_rrf(1 / 61, 2) == pytest.approx(0.5)
    # a near-tie must stay a near-tie (the flaw min-max scaling had)
    assert relevance_from_rrf(1 / 61, 1) - relevance_from_rrf(1 / 62, 1) < 0.02
    # rank 50 in a single list still scores well above zero
    assert 0.5 < relevance_from_rrf(1 / 110, 1) < 0.6
