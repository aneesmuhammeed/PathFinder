"""Render eval results as a Markdown report (EVALS.md)."""

from datetime import date

from evals.agent_eval import ScenarioResult, by_dimension
from evals.retrieval_eval import QueryResult, aggregate

METRICS = [("ndcg@10", "nDCG@10"), ("p@10", "P@10"), ("mrr", "MRR"), ("recall@50", "Recall@50")]
CATS = ["role_title", "skills", "paraphrase", "alias", "constrained", "personalised"]


def _table(headers: list[str], rows: list[list[str]], left: tuple[int, ...] = (0,)) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(":---" if i in left else "---:" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _split(results: list[QueryResult], split: str) -> list[QueryResult]:
    return [r for r in results if r.query.split == split]


def retrieval_section(abl: dict[str, list[QueryResult]], nq: dict) -> str:
    names = list(abl)
    overall = [[n, *[_f(aggregate(abl[n])["ALL"][m]) for m, _ in METRICS]] for n in names]
    by_cat = [[n, *[_f(aggregate(abl[n]).get(c, {}).get("ndcg@10")) for c in CATS]] for n in names]
    n_by_cat = aggregate(abl[names[0]])
    splits = [[n, _f(aggregate(_split(abl[n], "dev"))["ALL"]["ndcg@10"]), _f(aggregate(_split(abl[n], "test"))["ALL"]["ndcg@10"])] for n in names]
    pers_names = ["hybrid (RRF)", "hybrid + rerank (profile ignored)", "hybrid + rerank + profile"]
    pers = []
    for n in pers_names:
        rs = [r for r in abl[n] if r.query.category == "personalised" and r.metrics]
        topical = [r.topical_p10 for r in rs if r.topical_p10 is not None]
        pers.append([n, _f(aggregate(abl[n]).get("personalised", {}).get("ndcg@10")), _f(sum(topical) / len(topical) if topical else None)])
    return f"""## 1. Retrieval

{nq['dev'] + nq['test']} queries ({nq['dev']} dev / {nq['test']} test) over the 520-job labelled corpus, scored with graded relevance judgments for every job, using the *same code path as the API*.
One further query (`con-frontend-remote`) was added after a failure was seen in a demo; it is a canary, excluded from every number below except where it is named.
nDCG@10 uses linear gain; P@10 counts any relevant result (grade >= 1); MRR is the reciprocal rank of the first relevant result; Recall@50 is exact (all relevant jobs are known).

### 1.1 Ablation: what does each stage add?

{_table(['Configuration', *[m for _, m in METRICS]], overall)}

### 1.2 nDCG@10 by query category

{_table(['Configuration', *CATS], by_cat)}

Queries per category: {', '.join(f"{c} {int(n_by_cat[c]['n'])}" for c in CATS if c in n_by_cat)}. (`personalised` is judged on topical relevance x personal fit, so only the profile-aware row is expected to do well there.)
The `alias` queries were written from the skill taxonomy's own aliases (k8s, SRE, genai...), so the jump from alias expansion on that row is partly built in; the same feature also lifts `paraphrase`, which wasn't targeted.

### 1.3 Dev vs held-out test (nDCG@10)

{_table(['Configuration', 'dev', 'test'], splits)}

### 1.4 Personalisation

Judged on relevance x fit to the profile (seniority, job type, location, skills, work mode). "Topical P@10" re-scores the same results ignoring the profile, to check personalisation doesn't cost relevance.

{_table(['Configuration', 'nDCG@10 (personal fit)', 'Topical P@10'], pers)}

> The personal-fit judgments use the same signals the re-ranker uses, by construction. This table shows the re-ranker
> *does promote what the profile asks for without wrecking topical relevance*, not that those signals are the right ones.
"""


def failures_section(results: list[QueryResult], n: int = 8) -> str:
    scored = sorted((r for r in results if r.metrics), key=lambda r: r.metrics["ndcg@10"])[:n]
    rows = []
    for r in scored:
        top = "; ".join(f"{t} ({g:g})" for t, g in r.top[:3])
        note = " *(canary)*" if r.query.split == "canary" else ""
        rows.append([f"`{r.query.id}`{note}", f"{r.query.text or '(profile only)'}", _f(r.metrics["ndcg@10"]), top])
    return f"""### 1.5 Failure analysis: the {n} weakest queries (hybrid + rerank + profile)

{_table(['id', 'query', 'nDCG@10', 'top results (grade)'], rows, left=(0, 1, 3))}

What these show:
* **Paraphrases** (`para-*`) share no words with the postings. The default `hash` embedder is lexical, so it can't bridge that gap; that is exactly what a real
  semantic model (`EMBEDDER=fastembed`) is for, and the first thing to measure once it is available.
* **A vague query with a specific profile** (`pers-mobile-intern`: "engineering roles" + a mobile/Kotlin profile) scores 0: personalisation only *re-ranks* the 50
  candidates the query retrieves, and none were mobile. The fix is profile-aware retrieval (a third retriever seeded from the profile). Not built: with only five
  personalised queries there is too little data to tune it responsibly.
"""


def improvement_section(grid: list[dict], vweights: list[dict], sweep: list[dict], shipped: dict, baseline: dict) -> str:
    """`shipped` = the values the code actually uses: {"chars", "title_x", "vector_weight", "recency"}. Each knob's selection
    rule is re-applied to the measurements here; any disagreement with what ships is flagged loudly."""
    from evals.tuning import CANARY_GATE, canary_gate, pick_largest_within

    best = pick_largest_within(grid, "chars", "dev", 0.0)  # exact best on dev; ties go to more description text
    chosen_vw = pick_largest_within(vweights, "vector_weight", "dev", 0.005, canary_gate)["vector_weight"]
    dev_only_vw = pick_largest_within(vweights, "vector_weight", "dev", 0.005)
    for r in sweep:
        r["objective"] = (r["no_profile_dev"] + r["profile_dev"]) / 2
    prev = next(r for r in sweep if r["recency_weight"] == 0.07)
    chosen_recency = pick_largest_within(sweep, "recency_weight", "objective", 0.002)["recency_weight"]
    best_vw = max(vweights, key=lambda r: r["dev"])
    problems = []
    if (best["chars"], best["title_x"]) != (shipped["chars"], shipped["title_x"]):
        problems.append(f"recipe: rule picks {best['chars']} chars x{best['title_x']}, code ships {shipped['chars']} chars x{shipped['title_x']}")
    if chosen_vw != shipped["vector_weight"]:
        problems.append(f"vector weight: rule picks {chosen_vw:g}, code ships {shipped['vector_weight']:g}")
    if chosen_recency != shipped["recency"]:
        problems.append(f"recency weight: rule picks {chosen_recency:g}, code ships {shipped['recency']:g}")
    guard = ("> ⚠️ **Selection rules disagree with the shipped defaults: " + "; ".join(problems) + ".**\n" if problems
             else "> ✅ Consistency check: re-applying each selection rule to the measurements below reproduces the values the code ships.\n")
    vw_rows = [[f"{r['vector_weight']:g}" + (" (chosen)" if r["vector_weight"] == chosen_vw else " (equal vote)" if r["vector_weight"] == 1.0 else ""),
                _f(r["dev"]), _f(r["test"]), _f(r["all"]), _f(r["canary_ndcg"])] for r in vweights]

    def tag(r):
        key = (r["chars"], r["title_x"])
        return " (original)" if key == (2000, 1) else " (Phase 4 default)" if key == (300, 3) else " (chosen)" if r is best else ""

    grid_rows = [[f"{r['chars']} chars, title x{r['title_x']}{tag(r)}", _f(r["dev"]), _f(r["test"]), _f(r["all"])] for r in grid]
    sw_rows = [[f"{r['recency_weight']:g}" + (" (chosen)" if r["recency_weight"] == chosen_recency else " (previous default)" if r["recency_weight"] == 0.07 else ""),
                _f(r["no_profile_dev"]), _f(r["no_profile_test"]), _f(r["profile_dev"]), _f(r["profile_test"])] for r in sweep]
    return f"""## 2. Improvement log

Changes made because of measurements, selected on `dev` and confirmed on `test`. Includes one reversal of an earlier change of mine.
{guard}
### 2.1 Embedding text recipe for the keyless `hash` embedder (a correction)
The first recipe embedded 2000 chars of description with the title once. Feature hashing into 384 slots collides on long text, and "frontend engineer" returned backend jobs from
vector search. I "fixed" it by truncating the description to 300 chars and repeating the title 3x, judged by vector-only precision on 13 short queries. **On this broader eval set,
judged on the hybrid pipeline we actually ship, that was a net loss** (grid below): hybrid search was already compensating through the keyword retriever. A grid over both knobs, selected on dev:

{_table(['Recipe (hybrid RRF, non-personalised queries)', 'dev nDCG@10', 'test', 'all'], grid_rows)}

Chosen: **{best['chars']} chars, title x{best['title_x']}** (dev {_f(best['dev'])}, test {_f(best['test'])}), versus the Phase 4 default at test {_f(next(r for r in grid if (r['chars'], r['title_x']) == (300, 3))['test'])}.
(1000 and 2000 chars tie because few postings exceed 1000 chars.) Lesson recorded: tune the stage you ship, on a set you didn't tune on. Knobs are tuned one at a time (recipe, then fusion weight, then recency), not jointly.

### 2.2 How much should the vector retriever's vote count? (weighted RRF)
With the recipe above, one visible query still failed: "frontend engineer" (remote only) interleaved backend jobs, because the lexical hash vector got an equal vote with keyword
search in the fusion. Reciprocal Rank Fusion takes per-retriever weights; sweeping the vector weight (recipe fixed; keyword = 1.0).
The canary column is that exact query (`con-frontend-remote`), added to the set *after* the failure was seen, so it is a sanity check, not independent evidence:

{_table(['Vector weight', 'dev nDCG@10', 'test', 'all', 'nDCG on the canary'], vw_rows)}

**Selection rule (two parts):** the largest weight within 0.005 of the best dev score (best: {best_vw['vector_weight']:g} at {_f(best_vw['dev'])}) **that also passes a regression gate**: the canary must reach nDCG@10 >= {CANARY_GATE}.
Dev alone cannot separate these settings (they are within noise of each other), and the dev-only winner ({dev_only_vw['vector_weight']:g}) leaves the known failure at {_f(dev_only_vw['canary_ndcg'])}, so the gate breaks the tie. That is a deliberate product judgement ("a failure I have seen must not ship"),
and it costs {_f(dev_only_vw['dev'] - next(r for r in vweights if r['vector_weight'] == chosen_vw)['dev'])} dev nDCG. Chosen: **{chosen_vw:g}**. Test nDCG is {_f(next(r for r in vweights if r['vector_weight'] == chosen_vw)['test'])} there versus {_f(next(r for r in vweights if r['vector_weight'] == 1.0)['test'])} at an equal vote.
The vector side keeps a small vote because it still adds roughly 0.025 nDCG over keyword search alone.
*Correction:* a quick probe with a candidate pool of 10 suggested 0.75 fixed the canary; at the real default pool size of 50 it does not ({_f(next(r for r in vweights if r['vector_weight'] == 0.75)['canary_ndcg'])}).
This is a property of the *hash* embedder (`HashEmbedder.rrf_weight`); a real semantic model should be re-tuned from 1.0.

### 2.3 Re-ranker recency weight
With no profile, re-ranking should be roughly neutral. At the previous default (0.07) it was not: on dev it scored {_f(prev['no_profile_dev'])} against {_f(baseline['dev'])} for plain hybrid, because the recency feature reshuffled equally relevant jobs
(on test the two were within noise: {_f(prev['no_profile_test'])} vs {_f(baseline['test'])}, so this is a modest effect and a small change). The sweep picks the weight
on the **dev** split only (selection rule: the largest recency weight within 0.002 of the best dev objective, so freshness still counts a little); **test** is reported alongside.

{_table(['Recency weight', 'no profile: dev', 'test', 'with profile: dev', 'test'], sw_rows)}
"""


def agent_section(results: list[ScenarioResult], provider: str, controls: dict[str, list[ScenarioResult]], judged: bool) -> str:
    is_mock = provider.startswith("mock")
    n_ok = sum(r.ok for r in results)
    dims = by_dimension(results)
    dim_rows = [[d, f"{a}/{b}", f"{100 * a / b:.0f}%"] for d, (a, b) in sorted(dims.items())]
    sc_rows = []
    for r in results:
        bad = [n for n, ok in r.passed.items() if not ok]
        sc_rows.append([f"`{r.scenario.id}`", "✅" if r.ok else "❌", str(r.run.steps), f"{r.run.input_tokens + r.run.output_tokens}",
                        f"{r.run.seconds:.1f}s", "; ".join(bad) or ""])
    ctrl_rows = [[name, f"{sum(x.ok for x in rs)}/{len(rs)}"] for name, rs in controls.items()]
    judge_line = ""
    if judged:
        v = [r.judge.verdict for r in results if r.judge]
        scored = [x for x in v if x]
        judge_line = (f"\nLLM judge (same provider, so biased toward its own answers): {sum(x.supported for x in scored)}/{len(scored)} answers judged fully "
                      f"supported by tool outputs; {len(v) - len(scored)} unscored.\n")
    num_bad = [(r.scenario.id, r.ungrounded_numbers) for r in results if r.ungrounded_numbers]
    banner = ("> **This run used the built-in rule-based mock, so it is a regression test of the agent plumbing (tool dispatch, memory, grounding, "
              "streaming), not a measurement of language-model quality.** The mock and the checks were written by the same author, so a high pass rate "
              "here is expected. Re-run with a real provider for the numbers that matter: `python -m evals.run --agent --llm auto --allow-paid`.\n") if is_mock else \
             f"> Run with a real provider (`{provider}`). Single run per scenario: LLM behaviour is stochastic, so treat one failure as a signal to re-run, not a verdict.\n"
    return f"""## 3. Agent

Provider: **{provider}**. {len(results)} scenarios, {sum(len(r.passed) for r in results)} checks.

{banner}
**{n_ok}/{len(results)} scenarios passed (all checks).**

### 3.1 By dimension

{_table(['Dimension', 'Checks passed', 'Rate'], dim_rows)}

### 3.2 Scenarios

{_table(['Scenario', 'Pass', 'Steps', 'Tokens', 'Time', 'Failed checks'], sc_rows)}

Avg steps {sum(r.run.steps for r in results) / len(results):.1f} | avg tokens {sum(r.run.input_tokens + r.run.output_tokens for r in results) // len(results)} | avg latency {sum(r.run.seconds for r in results) / len(results):.1f}s per scenario.
Salary figures not traceable to tool output: {num_bad or 'none'}.
{judge_line}
### 3.3 Negative controls: can the checks tell a bad agent from a good one?

{_table(['Agent', 'Scenarios passed'], [[provider + ' (system under test)', f'{n_ok}/{len(results)}'], *ctrl_rows])}

The *lazy* control never calls a tool; the *gullible* control obeys instructions embedded in job descriptions and invents jobs and salaries.
A trustworthy suite scores them far below the real agent. The couple of scenarios they still pass are ones where doing nothing is the correct behaviour
(e.g. `out-of-scope`, `sensitive-data`), which is a reminder that safety checks alone can be satisfied by an agent that does nothing, so they are always read alongside the tool-use checks.
"""


def render(*, retrieval: str | None, failures: str | None, improvement: str | None, agent: str | None, setup: dict) -> str:
    head = f"""# Evaluation report

Generated {date.today().isoformat()} by `python -m evals.run`. Embedder: **{setup['embedder']}** | corpus: {setup['corpus']} synthetic jobs | agent provider: **{setup['provider']}**.

**How to read this.** The corpus is synthetic and generated from the same role/skill taxonomy the judgments use, so these numbers are an upper bound and a
*regression harness*: they show whether a change helps or hurts, and which kinds of query break, not how the product would score on the real job market.
Tuning was done on the `dev` split only; `test` is reported alongside it.
"""
    return "\n".join(s for s in (head, retrieval, failures, improvement, agent) if s)
