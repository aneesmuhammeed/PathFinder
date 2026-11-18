import json
from collections import Counter
from pathlib import Path

import pytest

from app.db.models import JOB_TYPES, SENIORITY, WORK_MODES

SEED = Path(__file__).resolve().parents[2] / "data" / "seed_jobs.jsonl"


@pytest.fixture(scope="module")
def jobs():
    return [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines()]


def test_seed_size_and_unique_ids(jobs):
    assert len(jobs) >= 500
    assert len({j["external_id"] for j in jobs}) == len(jobs)


def test_seed_fields_match_schema_enums(jobs):
    for j in jobs:
        assert j["seniority"] in SENIORITY
        assert j["job_type"] in JOB_TYPES
        assert j["work_mode"] in WORK_MODES
        assert j["title"] and len(j["description"]) > 200 and j["skills"]
        if j["salary_min"] is not None:
            assert j["salary_max"] >= j["salary_min"]
            assert j["salary_currency"] and j["salary_period"] in ("year", "month")


def test_seed_is_diverse_and_india_heavy(jobs):
    assert len({j["role_family"] for j in jobs}) >= 10
    india = sum(j["country"] == "India" for j in jobs) / len(jobs)
    assert india > 0.6
    assert Counter(j["job_type"] for j in jobs)["internship"] >= 50  # interns are a core audience
    assert {"Bengaluru", "Gurugram"} <= {j["location"] for j in jobs}


def test_seed_is_deterministic():
    import importlib.util
    import random

    spec = importlib.util.spec_from_file_location("gen", Path(__file__).resolve().parents[1] / "scripts" / "generate_seed.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    a = [gen.build_job(i, random.Random(42)) for i in range(1, 4)]
    b = [gen.build_job(i, random.Random(42)) for i in range(1, 4)]
    assert a == b
