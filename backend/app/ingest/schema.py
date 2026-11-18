"""The canonical in-memory shape every source adapter must produce."""

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, field_validator

from app.db.models import JOB_TYPES, SENIORITY, WORK_MODES
from app.ingest.skills import canonicalize


class NormalizedJob(BaseModel):
    source: str
    external_id: str
    company: str
    title: str
    description: str
    url: str | None = None
    location: str | None = None
    country: str | None = None
    work_mode: str = "onsite"
    job_type: str = "full_time"
    seniority: str = "mid"
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    posted_at: datetime | None = None
    skills: list[str] = []

    @field_validator("company", "title", "external_id", "description", mode="before")
    @classmethod
    def _strip_required(cls, v):
        v = (v or "").strip() if isinstance(v, str) else str(v)
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("company")
    @classmethod
    def _company_len(cls, v):
        return v[:200]

    @field_validator("title")
    @classmethod
    def _title_len(cls, v):
        return v[:300]

    @field_validator("description")
    @classmethod
    def _desc_len(cls, v):
        return v[:20000]

    @field_validator("location")
    @classmethod
    def _loc_len(cls, v):
        return v[:200] if v else None

    @field_validator("url")
    @classmethod
    def _url_len(cls, v):
        return v[:1000] if v else None

    @field_validator("seniority", mode="before")
    @classmethod
    def _seniority(cls, v):
        return v if v in SENIORITY else "mid"

    @field_validator("job_type", mode="before")
    @classmethod
    def _job_type(cls, v):
        return v if v in JOB_TYPES else "full_time"

    @field_validator("work_mode", mode="before")
    @classmethod
    def _work_mode(cls, v):
        return v if v in WORK_MODES else "onsite"

    @field_validator("skills", mode="before")
    @classmethod
    def _skills(cls, v):
        seen: dict[str, None] = {}
        for s in v or []:
            c = canonicalize(str(s))
            if c:
                seen.setdefault(c[:100], None)
        return list(seen)

    def content_hash(self) -> str:
        """Hash of everything that matters for search. `posted_at` is deliberately excluded:
        a changing timestamp alone must not trigger an update or a re-embed."""
        payload = {
            "company": self.company, "title": self.title, "description": self.description, "url": self.url,
            "location": self.location, "country": self.country, "work_mode": self.work_mode,
            "job_type": self.job_type, "seniority": self.seniority,
            "salary": [self.salary_min, self.salary_max, self.salary_currency, self.salary_period],
            "skills": sorted(self.skills),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
