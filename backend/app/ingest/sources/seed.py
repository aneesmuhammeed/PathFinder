"""Offline seed corpus (data/seed_jobs.jsonl). Guarantees the project runs with no network."""

import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.ingest.schema import NormalizedJob

DEFAULT_PATH = Path(__file__).resolve().parents[4] / "data" / "seed_jobs.jsonl"


class SeedSource:
    name = "seed"
    snapshot = True
    min_interval = None

    def __init__(self, path: Path | str | None = None, today: datetime | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        # Anchor to midnight UTC so "N days ago" is stable within a day (re-runs stay idempotent).
        now = today or datetime.now(timezone.utc)
        self.anchor = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def fetch(self) -> Iterable[NormalizedJob]:
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                yield NormalizedJob(
                    source=self.name,
                    external_id=r["external_id"],
                    company=r["company"],
                    title=r["title"],
                    description=r["description"],
                    url=r.get("url"),
                    location=r.get("location"),
                    country=r.get("country"),
                    work_mode=r.get("work_mode"),
                    job_type=r.get("job_type"),
                    seniority=r.get("seniority"),
                    salary_min=r.get("salary_min"),
                    salary_max=r.get("salary_max"),
                    salary_currency=r.get("salary_currency"),
                    salary_period=r.get("salary_period"),
                    posted_at=self.anchor - timedelta(days=r.get("posted_days_ago", 0)),
                    skills=r.get("skills", []),
                )
