"""Remotive public API (https://remotive.com/api/remote-jobs).

Terms (from the API's own legal notice): link back to the Remotive URL and credit Remotive as the
source, don't push jobs to other job boards, and poll sparingly (they advise max ~4 times a day).
We therefore make ONE request per run, enforce a minimum interval, keep `url` pointing at Remotive,
and the UI must show "via Remotive".
"""

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

import httpx

from app.core.http import SourceError, get_json
from app.ingest.schema import NormalizedJob
from app.ingest.skills import extract_skills, is_known
from app.ingest.text import html_to_text, infer_country, infer_seniority, is_tech_role, parse_salary

log = logging.getLogger("pathfinder.ingest.remotive")

API_URL = "https://remotive.com/api/remote-jobs"
MAX_SKILLS = 12
MAX_TRUSTED_TAGS = 10
TECH_CATEGORY_HINTS = ("software", "data", "artificial intelligence", "information technology", "devops", "qa",
                       "product", "design", "machine learning", "engineering")
JOB_TYPES = {"full_time": "full_time", "contract": "contract", "freelance": "contract", "part_time": "part_time",
             "internship": "internship"}


class RemotiveSource:
    name = "remotive"
    snapshot = True
    min_interval = timedelta(hours=6)

    def __init__(self, client: httpx.Client | None = None, url: str = API_URL):
        self.client = client
        self.url = url

    def fetch(self) -> Iterable[NormalizedJob]:
        data = get_json(self.url, client=self.client)
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
            raise SourceError("remotive: unexpected response shape (no 'jobs' list)")
        out = []
        for raw in data["jobs"]:
            try:
                job = self._normalize(raw)
            except (KeyError, ValueError, TypeError) as exc:  # one bad row must not sink the run
                log.warning("skipping malformed remotive job", extra={"id": raw.get("id") if isinstance(raw, dict) else None, "err": str(exc)})
                continue
            if job:
                out.append(job)
        return out

    def _normalize(self, raw: dict) -> NormalizedJob | None:
        category = (raw.get("category") or "").lower()
        if not any(h in category for h in TECH_CATEGORY_HINTS):
            return None
        title = raw["title"]
        description = html_to_text(raw.get("description") or "")
        tags = [t for t in (raw.get("tags") or []) if isinstance(t, str) and is_known(t)]
        if len(tags) > MAX_TRUSTED_TAGS:  # a poster who tags everything has told us nothing
            tags = []
        skills = list(dict.fromkeys([*tags, *extract_skills(f"{title}\n{description}", title=title, limit=MAX_SKILLS)]))[:MAX_SKILLS]
        if not is_tech_role(title, skills):
            return None
        sal = parse_salary(raw.get("salary"))
        location = (raw.get("candidate_required_location") or "Worldwide").strip()
        posted = raw.get("publication_date")
        posted_at = datetime.fromisoformat(posted).replace(tzinfo=timezone.utc) if posted else None
        return NormalizedJob(
            source=self.name,
            external_id=str(raw["id"]),
            company=raw["company_name"],
            title=title,
            description=description,
            url=raw.get("url"),
            location=f"Remote ({location})",
            country=infer_country(location),
            work_mode="remote",
            job_type=JOB_TYPES.get(raw.get("job_type") or "", "full_time"),
            seniority=infer_seniority(title),
            salary_min=sal[0] if sal else None,
            salary_max=sal[1] if sal else None,
            salary_currency=sal[2] if sal else None,
            salary_period=sal[3] if sal else None,
            posted_at=posted_at,
            skills=skills,
        )
