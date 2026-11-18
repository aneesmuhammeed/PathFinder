"""Second-stage, profile-aware re-ranking.

A transparent weighted sum of features, each in [0, 1]. Deliberately NOT another LLM call:
it's fast, free, deterministic, testable, and every point of the score can be explained to the user
("matches 4 of your skills", "seniority is one level above yours").

* Features that can't be computed (the user never told us their seniority, the job has no salary)
  are *omitted* and the remaining weights are renormalised, so a sparse profile isn't penalised.
* Query relevance is the fused RRF score divided by its theoretical maximum (rank 1 in every retriever),
  so it's an *absolute* 0..1 measure. (Min-max scaling over the pool was tried and rejected: it blows a
  meaningless tie between two near-identical jobs up into a full 0..1 gap that swamps personalisation.)
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.db.models import SENIORITY, Job
from app.search.schema import ProfileSignals, ScoreComponent

# Relative importance. Not required to sum to 1: they are normalised over the features that can be assessed
# for each result. `recency` was 0.07 until the eval harness showed it reshuffled equally relevant jobs
# (EVALS.md 2.3, selected on dev: no-profile nDCG +0.010 dev / +0.003 test going 0.07 -> 0.04); freshness stays a tie-breaker.
WEIGHTS = {
    "relevance": 0.36,
    "skills": 0.20,
    "seniority": 0.09,
    "job_type": 0.06,
    "work_mode": 0.05,
    "location": 0.08,
    "salary": 0.04,
    "recency": 0.04,
    "feedback": 0.05,
}
RECENCY_HALF_LIFE_DAYS = 21.0
SKILL_COVERAGE_CAP = 8  # a posting listing 14 skills shouldn't need 14 matches to look good


@dataclass
class Affinity:
    """What the user's saves/dismissals say they like (built from `job_feedback`)."""

    liked_skills: Counter = field(default_factory=Counter)
    disliked_skills: Counter = field(default_factory=Counter)
    liked_companies: set[str] = field(default_factory=set)
    disliked_companies: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.liked_skills or self.disliked_skills or self.liked_companies or self.disliked_companies)


def build_affinity(saved: list[Job], dismissed: list[Job]) -> Affinity:
    a = Affinity()
    for j in saved:
        a.liked_skills.update(s.name for s in j.skills)
        a.liked_companies.add(j.company.name)
    for j in dismissed:
        a.disliked_skills.update(s.name for s in j.skills)
        a.disliked_companies.add(j.company.name)
    return a


def relevance_from_rrf(rrf: float, max_weight: float, k: int = 60) -> float:
    """1.0 = ranked first by every retriever; 0.5 = ranked first by one of two equally-weighted; ~0.55 for rank 50
    in one list. `max_weight` is the sum of the active retrievers' weights (2.0 for plain two-way hybrid)."""
    return min(1.0, rrf * (k + 1) / max(1e-9, max_weight))


def _level(s: str) -> int:
    return SENIORITY.index(s)


def _age_days(job: Job, now: datetime) -> float | None:
    if job.posted_at is None:
        return None
    posted = job.posted_at if job.posted_at.tzinfo else job.posted_at.replace(tzinfo=timezone.utc)
    return max(0.0, (now - posted).total_seconds() / 86400)


def _annual(amount: int, period: str | None) -> int:
    return amount * 12 if period == "month" else amount


# Each feature returns (value 0..1, detail) or None when it can't be assessed.
Feature = tuple[float, str] | None


def f_skills(job: Job, p: ProfileSignals | None) -> Feature:
    if not p or not p.skills:
        return None
    job_skills = [s.name for s in job.skills]
    if not job_skills:
        return None
    have = [s for s in job_skills if s in set(p.skills)]
    denom = min(len(job_skills), SKILL_COVERAGE_CAP)
    value = min(1.0, len(have) / denom)
    shown = ", ".join(have[:5]) if have else "none"
    lead = "only " if value < 0.5 else ""
    return value, f"{lead}{len(have)} of {len(job_skills)} listed skills match yours ({shown})"


def f_seniority(job: Job, p: ProfileSignals | None) -> Feature:
    if not p or not p.seniority:
        return None
    gap = _level(job.seniority) - _level(p.seniority)
    value = {0: 1.0, 1: 0.5, -1: 0.5}.get(gap, 0.0)
    if gap == 0:
        return value, f"{job.seniority} level matches yours"
    direction = "above" if gap > 0 else "below"
    return value, f"{job.seniority} is {abs(gap)} level(s) {direction} your {p.seniority}"


def f_job_type(job: Job, p: ProfileSignals | None) -> Feature:
    if not p or not p.job_type:
        return None
    ok = job.job_type == p.job_type
    return (1.0 if ok else 0.0), (f"{job.job_type.replace('_', ' ')} as you want" if ok else f"it's {job.job_type.replace('_', ' ')}, you want {p.job_type.replace('_', ' ')}")


def f_work_mode(job: Job, p: ProfileSignals | None) -> Feature:
    if not p or not p.work_mode:
        return None
    if job.work_mode == p.work_mode:
        return 1.0, f"{job.work_mode} as you prefer"
    if "hybrid" in (job.work_mode, p.work_mode):
        return 0.5, f"{job.work_mode} vs your preferred {p.work_mode}"
    return 0.0, f"{job.work_mode}, you prefer {p.work_mode}"


def f_location(job: Job, p: ProfileSignals | None) -> Feature:
    if not p or not p.preferred_locations:
        return None
    hay = f"{job.location or ''} {job.country or ''}".lower()
    for loc in p.preferred_locations:
        if loc.strip().lower() in hay:
            return 1.0, f"in {loc}"
    if job.work_mode == "remote":
        return 0.6, "remote, so location doesn't limit you"
    return 0.0, f"{job.location or 'unknown location'} isn't in your preferred locations"


def f_salary(job: Job, p: ProfileSignals | None) -> Feature:
    if not p or not p.min_salary or not job.salary_max or job.salary_currency != "INR":
        return None
    top = _annual(job.salary_max, job.salary_period)
    if top >= p.min_salary:
        return 1.0, "pays at or above your minimum"
    return (top / p.min_salary) ** 2, "tops out below your minimum"


def f_recency(job: Job, now: datetime) -> Feature:
    age = _age_days(job, now)
    if age is None:
        return 0.3, "posting date unknown"
    return 0.5 ** (age / RECENCY_HALF_LIFE_DAYS), ("posted today" if age < 1 else f"posted {int(age)} day(s) ago")


def f_feedback(job: Job, a: Affinity | None) -> Feature:
    if not a:
        return None
    names = [s.name for s in job.skills]
    delta = sum(a.liked_skills[s] for s in names) - sum(a.disliked_skills[s] for s in names)
    value = 0.5 + 0.5 * math.tanh(delta / 3.0)
    company = job.company.name
    if company in a.liked_companies:
        value += 0.25
    if company in a.disliked_companies:
        value -= 0.25
    value = min(1.0, max(0.0, value))
    note = "similar to jobs you saved" if value > 0.6 else "similar to jobs you dismissed" if value < 0.4 else "no strong signal from your saves"
    return value, note


def score_job(
    job: Job, relevance: float, profile: ProfileSignals | None, affinity: Affinity | None, now: datetime
) -> tuple[float, list[ScoreComponent], list[str], list[str]]:
    raw: dict[str, Feature] = {
        "relevance": (relevance, "matches your search"),
        "skills": f_skills(job, profile),
        "seniority": f_seniority(job, profile),
        "job_type": f_job_type(job, profile),
        "work_mode": f_work_mode(job, profile),
        "location": f_location(job, profile),
        "salary": f_salary(job, profile),
        "recency": f_recency(job, now),
        "feedback": f_feedback(job, affinity),
    }
    present = {k: v for k, v in raw.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in present)
    components, reasons, caveats = [], [], []
    score = 0.0
    for name, (value, detail) in present.items():
        w = WEIGHTS[name] / total_w
        score += w * value
        components.append(ScoreComponent(feature=name, value=round(value, 3), weight=round(w, 3),
                                         contribution=round(w * value, 3), detail=detail))
        if name in ("relevance", "recency"):
            continue  # always present; not interesting to call out
        if value >= 0.75:
            reasons.append(detail)
        elif value <= 0.25:
            caveats.append(detail)
    return score, components, reasons, caveats
