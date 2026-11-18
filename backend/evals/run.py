"""CLI: python -m evals.run [--retrieval] [--agent] [--llm mock|auto] [--judge] [--embedder hash|fastembed] [--out PATH]

Defaults to everything that is free: retrieval evals + the agent suite against the built-in mock LLM.
A real provider costs money, so it needs --allow-paid.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from app.core.config import get_settings
from app.llm import get_llm_client
from app.llm.mock import MockLLM
from app.search import rerank
from app.search.embedder import get_embedder
from evals import report
from evals.agent_eval import run_agent_eval
from evals.agent_set import build_scenarios
from evals.controls import GullibleLLM, LazyLLM
from evals.env import eval_env
from evals.retrieval_eval import aggregate, run_ablation
from evals.retrieval_set import build_queries
from evals.tuning import recency_sweep, recipe_grid, vector_weight_sweep

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "EVALS.md"


def main(argv: list[str] | None = None) -> int:
    logger = logging.getLogger("pathfinder")
    previous = logger.level
    logger.setLevel(logging.ERROR)  # the negative controls trigger warnings on purpose
    try:
        return _main(argv)
    finally:
        logger.setLevel(previous)  # a CLI entry point must not leak global state (it is also called from tests)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.run")
    p.add_argument("--retrieval", action="store_true", help="run only the retrieval evals")
    p.add_argument("--agent", action="store_true", help="run only the agent evals")
    p.add_argument("--llm", choices=["mock", "auto"], default="mock", help="auto = provider from env keys (costs money)")
    p.add_argument("--allow-paid", action="store_true", help="required for --llm auto with a real provider")
    p.add_argument("--judge", action="store_true", help="also run the LLM groundedness judge (extra paid calls)")
    p.add_argument("--scenarios", help="comma-separated scenario ids")
    p.add_argument("--embedder", choices=["hash", "fastembed"], default=None)
    p.add_argument("--database-url", help="evaluate against this (disposable!) database instead of an embedded one")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--json", help="also dump raw numbers as JSON")
    a = p.parse_args(argv)
    do_retrieval, do_agent = (a.retrieval or not a.agent), (a.agent or not a.retrieval)

    settings = get_settings()
    if a.embedder:
        settings = settings.model_copy(update={"embedder": a.embedder})
    embedder = get_embedder(settings)

    llm = MockLLM() if a.llm == "mock" else get_llm_client(settings)
    if do_agent and llm.name != "mock" and not a.allow_paid:
        print(f"Refusing to call {llm.name} ({llm.model}) without --allow-paid: the agent suite makes roughly 40-80 model calls "
              f"(more with --judge). Re-run with --allow-paid, or use --llm mock.", file=sys.stderr)
        return 2
    scenarios = build_scenarios()
    if a.scenarios:
        wanted = set(a.scenarios.split(","))
        scenarios = [s for s in scenarios if s.id in wanted]

    retrieval = failures = improvement = agent = None
    raw: dict = {}
    with eval_env(embedder, a.database_url) as env:
        if do_retrieval:
            queries = build_queries()
            print(f"retrieval: {len(queries)} queries x 6 configurations ...", flush=True)
            abl = run_ablation(env, queries)
            recipe, vweights = [], []
            if embedder.name == "hash":  # recipe + fusion weight only exist for the hash embedder
                recipe = recipe_grid(env, queries)
                env.use_embedder(embedder)  # restore the shipped configuration for everything that follows
                vweights = vector_weight_sweep(env, queries)
            sweep = recency_sweep(env, queries, values=(0.07, 0.04, 0.02, 0.0))
            nq = {"total": len(queries), "dev": sum(q.split == "dev" for q in queries), "test": sum(q.split == "test" for q in queries)}
            retrieval = report.retrieval_section(abl, nq)
            failures = report.failures_section(abl["hybrid + rerank + profile"])
            shipped = {"chars": embedder.doc_params["description_chars"], "title_x": embedder.doc_params["title_repeats"],
                       "vector_weight": embedder.rrf_weight, "recency": rerank.WEIGHTS["recency"]} if recipe else None
            base = abl["hybrid (RRF)"]
            baseline = {s: aggregate([r for r in base if r.query.split == s])["ALL"]["ndcg@10"] for s in ("dev", "test")}
            improvement = report.improvement_section(recipe, vweights, sweep, shipped, baseline) if recipe else None
            raw["retrieval"] = {n: aggregate(r) for n, r in abl.items()}
            raw["recency_sweep"], raw["recipe_grid"], raw["vector_weight_sweep"] = sweep, recipe, vweights

        if do_agent:
            print(f"agent: {len(scenarios)} scenarios with {llm.name} ({llm.model}) ...", flush=True)
            judge_llm = llm if (a.judge and llm.name != "mock") else None
            results = run_agent_eval(env, llm, scenarios, judge_llm)
            controls = {"lazy (never uses tools)": run_agent_eval(env, LazyLLM(), scenarios),
                        "gullible (obeys injections, invents jobs)": run_agent_eval(env, GullibleLLM(), scenarios)}
            provider = f"{llm.name} ({llm.model})"
            agent = report.agent_section(results, provider, controls, judged=judge_llm is not None)
            raw["agent"] = {r.scenario.id: r.passed for r in results}

    text = report.render(retrieval=retrieval, failures=failures, improvement=improvement, agent=agent,
                         setup={"embedder": embedder.name, "corpus": 520, "provider": f"{llm.name} ({llm.model})" if do_agent else "n/a"})
    Path(a.out).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
