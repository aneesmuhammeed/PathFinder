import json
from pathlib import Path

import httpx
import pytest

from app.core.http import SourceError, get_json
from app.ingest.sources import ArbeitnowSource, RemotiveSource, SeedSource

LONG_EN = " We are hiring and you will build the product with our team for the users of the platform."


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestSeed:
    def test_reads_all_valid_jobs(self):
        jobs = list(SeedSource().fetch())
        assert len(jobs) >= 500
        assert all(j.source == "seed" and j.skills and j.posted_at for j in jobs)

    def test_posted_at_stable_within_a_day(self):
        a = {j.external_id: j.posted_at for j in SeedSource().fetch()}
        b = {j.external_id: j.posted_at for j in SeedSource().fetch()}
        assert a == b

    def test_custom_path(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({"external_id": "x", "company": "C", "title": "T", "description": "D",
                                 "seniority": "bogus", "skills": ["Postgres"]}) + "\n", encoding="utf-8")
        (job,) = SeedSource(p).fetch()
        assert job.seniority == "mid"  # invalid enum falls back instead of crashing
        assert job.skills == ["postgresql"]


REMOTIVE = {"jobs": [
    {"id": 1, "url": "https://remotive.com/remote-jobs/x-1", "title": "Senior Python Engineer", "company_name": "Acme",
     "category": "Software Development", "tags": ["python", "docker", "fullstack"], "job_type": "full_time",
     "publication_date": "2026-11-14T20:33:53", "candidate_required_location": "Worldwide",
     "salary": "$120k - $150k", "description": "<p>Build APIs with <b>FastAPI</b> and Postgres." + LONG_EN + "</p>"},
    {"id": 2, "url": "u", "title": "Content Writer", "company_name": "Blog", "category": "Writing", "tags": [],
     "job_type": "full_time", "publication_date": "2026-11-14T10:00:00", "candidate_required_location": "USA",
     "salary": "", "description": "<p>Write posts.</p>"},
    {"id": 3, "title": "Broken row without company", "category": "Software Development"},
]}


class TestRemotive:
    def test_normalizes_filters_and_survives_bad_rows(self):
        src = RemotiveSource(client=mock_client(lambda r: httpx.Response(200, json=REMOTIVE)))
        jobs = list(src.fetch())
        assert [j.external_id for j in jobs] == ["1"]  # writer filtered by category, broken row skipped
        j = jobs[0]
        assert j.work_mode == "remote" and j.seniority == "senior" and j.job_type == "full_time"
        assert (j.salary_min, j.salary_max, j.salary_currency) == (120000, 150000, "USD")
        assert {"python", "docker", "fastapi", "postgresql"} <= set(j.skills)
        assert "fullstack" not in j.skills  # unknown tags are not trusted as skills
        assert "<" not in j.description
        assert j.url.startswith("https://remotive.com/")  # link back (their terms)

    def test_keyword_stuffed_posting_is_capped_and_tags_distrusted(self):
        every_tag = ["python", "java", "php", "ruby", "react", "angular", "ios", "swift", "rust", "scala", "flutter", "firebase"]
        body = "<p>" + " ".join(every_tag) + " python python sql sql sql." + LONG_EN + "</p>"
        raw = {**REMOTIVE["jobs"][0], "id": 9, "title": "Senior Data Engineer", "tags": every_tag, "description": body}
        src = RemotiveSource(client=mock_client(lambda r: httpx.Response(200, json={"jobs": [raw]})))
        (job,) = src.fetch()
        assert len(job.skills) <= 12
        assert "python" in job.skills and "sql" in job.skills  # frequent ones survive the cap

    def test_unexpected_shape_raises(self):
        src = RemotiveSource(client=mock_client(lambda r: httpx.Response(200, json={"oops": 1})))
        with pytest.raises(SourceError):
            list(src.fetch())

    def test_makes_one_request(self):
        calls = []
        src = RemotiveSource(client=mock_client(lambda r: (calls.append(r), httpx.Response(200, json=REMOTIVE))[1]))
        list(src.fetch())
        assert len(calls) == 1


def arb_job(slug, title, desc, **kw):
    return {"slug": slug, "company_name": "Firma", "title": title, "description": desc, "remote": False,
            "url": f"https://www.arbeitnow.com/jobs/{slug}", "tags": [], "job_types": ["Permanent"],
            "location": "Berlin", "created_at": 1794934851, **kw}


class TestArbeitnow:
    def test_filters_german_and_non_tech_and_paginates(self):
        german = "Wir suchen einen Entwickler und Sie arbeiten mit unserem Team für die Plattform und der Kunden. " * 3
        pages = {
            "1": {"data": [arb_job("a", "Backend Developer (m/w/d)", "<p>Python and Django." + LONG_EN + "</p>"),
                           arb_job("b", "Softwareentwickler", german),
                           arb_job("c", "Nurse", "<p>Care for patients." + LONG_EN + "</p>")],
                  "links": {"next": "x"}},
            "2": {"data": [arb_job("d", "Junior Data Engineer", "<p>SQL and Spark." + LONG_EN + "</p>", remote=True)],
                  "links": {"next": None}},
        }
        seen = []

        def handler(req):
            page = req.url.params["page"]
            seen.append(page)
            return httpx.Response(200, json=pages[page])

        jobs = list(ArbeitnowSource(client=mock_client(handler), pages=5).fetch())
        assert seen == ["1", "2"]  # stops when there's no next page
        assert [j.external_id for j in jobs] == ["a", "d"]
        d = jobs[1]
        assert d.work_mode == "remote" and d.seniority == "junior" and "spark" in d.skills
        assert jobs[0].country == "Germany" and jobs[0].posted_at.year >= 2026  # inferred from "Berlin"

    def test_country_is_not_guessed_when_location_unknown(self):
        page = {"data": [arb_job("z", "Backend Developer", "<p>Python." + LONG_EN + "</p>", location="Remote - France")],
                "links": {"next": None}}
        (job,) = ArbeitnowSource(client=mock_client(lambda r: httpx.Response(200, json=page)), pages=1).fetch()
        assert job.country is None

    def test_window_source_is_not_a_snapshot(self):
        assert ArbeitnowSource.snapshot is False and RemotiveSource.snapshot is True


class TestHttpHelper:
    def test_retries_then_succeeds(self):
        n = {"c": 0}

        def handler(req):
            n["c"] += 1
            return httpx.Response(503) if n["c"] < 3 else httpx.Response(200, json={"ok": True})

        sleeps = []
        assert get_json("http://x", client=mock_client(handler), sleep=sleeps.append) == {"ok": True}
        assert n["c"] == 3 and sleeps == [1.0, 2.0]  # exponential backoff

    def test_gives_up_after_retries(self):
        with pytest.raises(SourceError, match="after 3 attempts"):
            get_json("http://x", client=mock_client(lambda r: httpx.Response(500)), sleep=lambda s: None)

    def test_4xx_is_not_retried(self):
        n = {"c": 0}

        def handler(req):
            n["c"] += 1
            return httpx.Response(404)

        with pytest.raises(SourceError, match="HTTP 404"):
            get_json("http://x", client=mock_client(handler), sleep=lambda s: None)
        assert n["c"] == 1

    def test_network_error_is_retried(self):
        def handler(req):
            raise httpx.ConnectError("boom")

        with pytest.raises(SourceError, match="ConnectError"):
            get_json("http://x", client=mock_client(handler), sleep=lambda s: None)

    def test_invalid_json(self):
        with pytest.raises(SourceError, match="invalid JSON"):
            get_json("http://x", client=mock_client(lambda r: httpx.Response(200, text="<html>")))
