"""Small, disciplined tuning experiments: choose on the DEV split only, then confirm on TEST."""

from contextlib import contextmanager

from app.search import rerank
from evals.env import EvalEnv
from evals.retrieval_eval import ABLATION, aggregate, run_variant
from evals.retrieval_set import EvalQuery

RERANK_NO_PROFILE = ABLATION[4]
RERANK_PROFILE = ABLATION[5]


@contextmanager
def weights(**overrides: float):
    old = dict(rerank.WEIGHTS)
    rerank.WEIGHTS.update(overrides)
    try:
        yield
    finally:
        rerank.WEIGHTS.clear()
        rerank.WEIGHTS.update(old)


CANARY_GATE = 0.75  # a known, visible failure must not come back: nDCG@10 on the canary query


def pick_largest_within(rows: list[dict], key: str, score: str, tolerance: float, gate=None) -> dict:
    """Selection rule used throughout: of the settings whose dev score is within `tolerance` of the best
    (and that pass the optional regression `gate`), take the one with the largest `key` (more information / more of
    the feature kept). Ties resolve the same way. The best score is taken over ALL rows, gated or not."""
    best = max(r[score] for r in rows)
    pool = [r for r in rows if r[score] >= best - tolerance and (gate is None or gate(r))]
    return max(pool or [max(rows, key=lambda r: r[score])], key=lambda r: r[key])


def canary_gate(row: dict) -> bool:
    return row["canary_ndcg"] >= CANARY_GATE


def recipe_grid(env: EvalEnv, queries: list[EvalQuery], dim: int = 384,
                chars=(300, 600, 1000, 2000), reps=(1, 2, 3)) -> list[dict]:
    """Embedding-text recipe for the hash embedder, judged on the pipeline we actually ship (hybrid RRF,
    non-personalised queries). Selection criterion: dev nDCG@10; test is reported alongside."""
    from app.search.embedder import HashEmbedder

    plain = [q for q in queries if q.category != "personalised"]
    hybrid = ABLATION[3]
    rows = []
    for c in chars:
        for r in reps:
            emb = HashEmbedder(dim)
            emb.doc_params = {"description_chars": c, "title_repeats": r}  # instance override of the class default
            emb.rrf_weight = 1.0  # recipe is chosen at equal fusion weights; the fusion weight is tuned next (coordinate descent)
            env.use_embedder(emb)
            res = run_variant(env, plain, hybrid)
            rows.append({"chars": c, "title_x": r,
                         "dev": aggregate(_by_split(res, "dev"))["ALL"]["ndcg@10"],
                         "test": aggregate(_by_split(res, "test"))["ALL"]["ndcg@10"],
                         "all": aggregate(res)["ALL"]["ndcg@10"]})
    return rows


def _by_split(results, split):
    return [r for r in results if r.query.split == split]


def vector_weight_sweep(env: EvalEnv, queries: list[EvalQuery], values=(1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)) -> list[dict]:
    """How much should the hash embedder's ranking count in hybrid fusion? Dev picks; test confirms.
    `canary_ndcg` tracks the query that failed visibly in a demo ('frontend engineer', remote only)."""
    plain = [q for q in queries if q.category != "personalised"]
    hybrid = ABLATION[3]
    canary = next(q for q in queries if q.id == "con-frontend-remote")
    rows = []
    shipped = env.embedder.rrf_weight
    for w in values:
        env.embedder.rrf_weight = w
        res = run_variant(env, plain, hybrid)
        row = {"vector_weight": w, "all": aggregate(res)["ALL"]["ndcg@10"]}
        for split in ("dev", "test"):
            row[split] = aggregate(_by_split(res, split))["ALL"]["ndcg@10"]
        row["canary_ndcg"] = next(r for r in res if r.query.id == canary.id).metrics["ndcg@10"]
        rows.append(row)
    env.embedder.rrf_weight = shipped
    return rows


def recency_sweep(env: EvalEnv, queries: list[EvalQuery], values=(0.07, 0.04, 0.02, 0.0)) -> list[dict]:
    """For each recency weight: nDCG@10 of the two re-ranking variants, per split.
    `objective` (the selection criterion) is the mean of the two variants' DEV scores."""
    rows = []
    for w in values:
        with weights(recency=w):
            row: dict = {"recency_weight": w}
            for label, v in (("no_profile", RERANK_NO_PROFILE), ("profile", RERANK_PROFILE)):
                res = run_variant(env, queries, v)
                for split in ("dev", "test"):
                    sub = [r for r in res if r.query.split == split]
                    row[f"{label}_{split}"] = aggregate(sub)["ALL"]["ndcg@10"]
            row["objective"] = (row["no_profile_dev"] + row["profile_dev"]) / 2
            rows.append(row)
    return rows
