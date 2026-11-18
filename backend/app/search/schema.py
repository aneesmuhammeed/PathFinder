"""Request/response models for search, shared by the REST API and (later) the agent's tools."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.ingest.skills import canonicalize

Seniority = Literal["intern", "junior", "mid", "senior", "lead"]
JobType = Literal["full_time", "internship", "contract", "part_time"]
WorkMode = Literal["remote", "hybrid", "onsite"]
SearchMode = Literal["hybrid", "vector", "keyword"]


class SearchFilters(BaseModel):
    """Hard constraints (applied in SQL, before ranking)."""

    seniority: list[Seniority] | None = None
    job_type: list[JobType] | None = None
    work_mode: list[WorkMode] | None = None
    locations: list[str] | None = Field(None, description="match city/country, case-insensitive substring")
    include_remote: bool = Field(False, description="with `locations`: also keep fully-remote jobs")
    skills_any: list[str] | None = Field(None, description="job must require at least one of these")
    posted_within_days: int | None = Field(None, ge=1, le=365)
    company: str | None = None
    exclude_job_ids: list[int] | None = None

    @field_validator("skills_any")
    @classmethod
    def _canon(cls, v):
        return [canonicalize(s) for s in v] if v else v


class ProfileSignals(BaseModel):
    """What we know about the searcher; drives personalised re-ranking. All optional."""

    skills: list[str] = []
    target_roles: list[str] = []
    preferred_locations: list[str] = []
    seniority: Seniority | None = None
    work_mode: WorkMode | None = None
    job_type: JobType | None = None
    min_salary: int | None = Field(None, description="annual INR; compared against INR-denominated jobs only")

    @field_validator("skills")
    @classmethod
    def _canon(cls, v):
        return list(dict.fromkeys(canonicalize(s) for s in v))


class SearchRequest(BaseModel):
    query: str = ""
    filters: SearchFilters = SearchFilters()
    profile: ProfileSignals | None = Field(None, description="inline profile; overrides the stored one")
    user_id: uuid.UUID | None = None
    limit: int = Field(10, ge=1, le=50)
    mode: SearchMode = "hybrid"
    rerank: bool = True
    expand: bool = Field(True, description="add canonical skill names for aliases used in the query (k8s → kubernetes)")


class JobOut(BaseModel):
    id: int
    title: str
    company: str
    location: str | None
    country: str | None
    work_mode: str
    job_type: str
    seniority: str
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    salary_period: str | None
    posted_at: datetime | None
    url: str | None
    source: str
    skills: list[str]
    snippet: str


class ScoreComponent(BaseModel):
    feature: str
    value: float = Field(description="0..1 how well this job fits on this feature")
    weight: float = Field(description="normalised weight actually applied")
    contribution: float
    detail: str


class RetrievalInfo(BaseModel):
    vector_rank: int | None
    keyword_rank: int | None
    rrf: float


class ScoredJob(BaseModel):
    job: JobOut
    score: float = Field(description="0..1 final score")
    relevance: float = Field(description="0..1 query relevance within the candidate pool")
    components: list[ScoreComponent]
    reasons: list[str]
    caveats: list[str]
    saved: bool = False
    retrieval: RetrievalInfo


class SearchResponse(BaseModel):
    query_used: str
    expansions: list[str] = Field(default=[], description="canonical skill terms added to the query")
    mode: SearchMode
    reranked: bool
    personalised: bool
    candidates: int
    took_ms: float
    results: list[ScoredJob]
