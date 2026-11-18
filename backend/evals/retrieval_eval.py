"""Run the retrieval eval set through the REAL search service (same code path as the API) under different
configurations, and score the rankings against the graded judgments."""

from collections import defaultdict
from dataclasses import dataclass, field

from app.search.schema import ProfileSignals, SearchFilters, SearchRequest
from app.search.service import InvalidSearch, search_jobs
from evals import metrics
from evals.env import EvalEnv
from evals.retrieval_set import EvalQuery


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str = "hybrid"
    expand: bool = True
    rerank: bool = False
    use_profile: bool = False  # only meaningful with rerank=True


# Ablation ladder, in the order a reader should think about it.
ABLATION = [
    Variant("keyword only", mode="keyword", expand=False),
    Variant("keyword + alias expansion", mode="keyword"),
    Variant("vector only", mode="vector"),
    Variant("hybrid (RRF)", mode="hybrid"),
    Variant("hybrid + rerank (profile ignored)", rerank=True),
    Variant("hybrid + rerank + profile", rerank=True, use_profile=True),
]


@dataclass
class QueryResult:
    query: EvalQuery
    metrics: dict[str, float] | None  # None => not applicable (skipped)
    top: list[tuple[str, float]] = field(default_factory=list)  # (title, grade) of the top 5
    topical_p10: float | None = None
    ms: float = 0.0


def run_variant(env: EvalEnv, queries: list[EvalQuery], v: Variant) -> list[QueryResult]:
    all_grades = {q.id: [q.judge(r) for r in env.records.values()] for q in queries}
    out: list[QueryResult] = []
    for q in queries:
        profile = ProfileSignals(**q.profile) if (q.profile and v.use_profile) else None
        req = SearchRequest(query=q.text, filters=SearchFilters(**q.filters), profile=profile, limit=50,
                            mode=v.mode, rerank=v.rerank, expand=v.expand)
        try:
            res = search_jobs(env.session, env.embedder, req, now=env.now)
        except InvalidSearch:  # e.g. a recommendations-only query with the profile withheld
            out.append(QueryResult(q, None))
            continue
        recs = [env.id_to_record.get(r.job.id) for r in res.results]
        grades = [q.judge(r) if r else 0.0 for r in recs]
        topical = None
        if q.topical:
            topical = metrics.precision_at_k([q.topical(r) if r else 0.0 for r in recs], 10)
        out.append(QueryResult(
            q, metrics.summarize(grades, all_grades[q.id]),
            [(r.job.title, g) for r, g in zip(res.results[:5], grades[:5], strict=True)], topical, res.took_ms))
    return out


def aggregate(results: list[QueryResult], *, by: str = "category") -> dict[str, dict[str, float]]:
    """Mean metrics per category / split, plus 'ALL'. Skipped queries and the post-hoc canary are excluded."""
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in results:
        if r.metrics is None or r.query.split == "canary":
            continue
        key = getattr(r.query, by)
        groups[key].append(r.metrics)
        groups["ALL"].append(r.metrics)
    return {k: {**metrics.mean_metrics(rows), "n": len(rows)} for k, rows in groups.items()}


def run_ablation(env: EvalEnv, queries: list[EvalQuery]) -> dict[str, list[QueryResult]]:
    return {v.name: run_variant(env, queries, v) for v in ABLATION}

