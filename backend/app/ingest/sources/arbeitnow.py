"""Arbeitnow free job-board API (https://www.arbeitnow.com/api/job-board-api).

Rolling window of recent jobs (paginated, ~300/page, refreshed hourly). Heavily German and not all tech,
so we keep English-language tech roles only. Because it's a window, a job disappearing from it does NOT
mean it was filled; jobs expire after a grace period (see `JOB_STALE_AFTER_DAYS`). The API asks for a link back.
"""

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

import httpx

from app.core.config import get_settings
from app.core.http import SourceError, get_json
from app.ingest.schema import NormalizedJob
from app.ingest.skills import extract_skills
from app.ingest.text import html_to_text, infer_country, infer_job_type, infer_seniority, is_tech_role, looks_english

log = logging.getLogger("pathfinder.ingest.arbeitnow")

API_URL = "https://www.arbeitnow.com/api/job-board-api"
MAX_SKILLS = 12


class ArbeitnowSource:
    name = "arbeitnow"
    snapshot = False
    min_interval = timedelta(hours=1)

    def __init__(self, client: httpx.Client | None = None, url: str = API_URL, pages: int | None = None):
        self.client = client
        self.url = url
        self.pages = pages if pages is not None else get_settings().arbeitnow_pages

    def fetch(self) -> Iterable[NormalizedJob]:
        out: list[NormalizedJob] = []
        for page in range(1, self.pages + 1):
            data = get_json(self.url, params={"page": page}, client=self.client)
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise SourceError("arbeitnow: unexpected response shape (no 'data' list)")
            for raw in data["data"]:
                try:
                    job = self._normalize(raw)
                except (KeyError, ValueError, TypeError) as exc:
                    log.warning("skipping malformed arbeitnow job", extra={"err": str(exc)})
                    continue
                if job:
                    out.append(job)
            if not (data.get("links") or {}).get("next"):
                break
        return out

    def _normalize(self, raw: dict) -> NormalizedJob | None:
        title = raw["title"]
        description = html_to_text(raw.get("description") or "")
        if not looks_english(f"{title}\n{description}"):
            return None
        tags = [t for t in (raw.get("tags") or []) if isinstance(t, str)]
        skills = extract_skills(f"{title}\n{' '.join(tags)}\n{description}", title=title, limit=MAX_SKILLS)
        if not is_tech_role(title, skills):
            return None
        job_types = [str(t) for t in (raw.get("job_types") or [])]
        hints = " ".join(job_types)
        location = (raw.get("location") or "").strip() or None
        created = raw.get("created_at")
        return NormalizedJob(
            source=self.name,
            external_id=raw["slug"],
            company=raw["company_name"],
            title=title,
            description=description,
            url=raw.get("url"),
            location=location,
            country=infer_country(location),  # never guess: the board is German but many jobs aren't
            work_mode="remote" if raw.get("remote") else "onsite",
            job_type=infer_job_type(title, hints),
            seniority=infer_seniority(title, hints),
            posted_at=datetime.fromtimestamp(int(created), tz=timezone.utc) if created else None,
            skills=skills,
        )
