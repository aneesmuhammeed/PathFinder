"""The labelled corpus: data/seed_jobs.jsonl. Every record carries generator-assigned ground truth
(`role_family`, `skills`, `seniority`, ...) that exists independently of the free text the search engine sees."""

import json
from pathlib import Path

SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "seed_jobs.jsonl"


def load_corpus(path: Path = SEED_PATH) -> dict[str, dict]:
    """external_id -> record."""
    with path.open(encoding="utf-8") as f:
        return {r["external_id"]: r for r in map(json.loads, filter(str.strip, f))}
