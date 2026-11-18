"""Tests for the evaluation harness itself. A harness nobody tests can quietly produce nonsense numbers."""

import math

import pytest

from app.llm.types import LLMError
from app.search.hybrid import rrf_fuse
from app.search.rerank import relevance_from_rrf
from app.search.schema import SearchFilters
from evals import metrics
from evals.agent_set import SECRET, build_scenarios
from evals.controls import GullibleLLM, LazyLLM
from evals.corpus import load_corpus
from evals.judge import JUDGE_SYSTEM, judge_answer, parse_verdict, ungrounded_numbers
from evals.retrieval_set import RELATED, build_queries, family, only_if, personalised, skills
from tests.helpers import ScriptedLLM, say


class TestMetrics:
    def test_ndcg_is_one_for_the_ideal_ranking_and_zero_for_nothing(self):
        all_grades = [3, 3, 1, 0, 0, 0]
        assert metrics.ndcg_at_k([3, 3, 1], all_grades, 10) == pytest.approx(1.0)
        assert metrics.ndcg_at_k([0, 0, 0], all_grades, 10) == 0.0
        assert metrics.ndcg_at_k([1, 3, 3], all_grades, 10) < 1.0  # order matters

    def test_ndcg_matches_a_hand_computation(self):
        ideal = 3 / math.log2(2) + 1 / math.log2(3)
        got = 1 / math.log2(2) + 3 / math.log2(3)
        assert metrics.ndcg_at_k([1, 3], [3, 1], 10) == pytest.approx(got / ideal)

    def test_no_relevant_documents_is_defined_not_a_crash(self):
        assert metrics.ndcg_at_k([0], [0, 0], 10) == 0.0 and metrics.recall_at_k([0], [0, 0]) == 0.0

    def test_precision_counts_empty_slots_as_misses(self):
        assert metrics.precision_at_k([3, 3], 10) == 0.2

    def test_mrr_and_recall(self):
        assert metrics.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3) and metrics.reciprocal_rank([0, 0]) == 0.0
        assert metrics.recall_at_k([1, 0, 1], [1, 1, 1, 1], k=3) == 0.5

    def test_mean(self):
        assert metrics.mean_metrics([{"a": 1.0}, {"a": 0.0}]) == {"a": 0.5} and metrics.mean_metrics([]) == {}


class TestJudgments:
    rec = {"role_family": "backend", "skills": ["python", "sql", "docker"], "seniority": "intern", "job_type": "internship",
           "location": "Pune", "work_mode": "remote"}

    def test_family_and_skills(self):
        assert family({"backend"})(self.rec) == 3
        assert family({"frontend"}, RELATED["frontend"])(self.rec) == 0
        assert family({"fullstack"}, RELATED["fullstack"])(self.rec) == 1  # backend is "related" to a fullstack query
        assert skills(["python", "sql"])(self.rec) == 3 and skills(["python", "react"])(self.rec) == 1 and skills(["react"])(self.rec) == 0

    def test_constraint_zeroes_out_violators(self):
        j = only_if(family({"backend"}), lambda r: r["location"] == "Pune")
        assert j(self.rec) == 3 and j({**self.rec, "location": "Delhi"}) == 0

    def test_personal_fit_scales_topical_relevance(self):
        profile = {"seniority": "intern", "job_type": "internship", "preferred_locations": ["Pune"], "skills": ["python", "sql"]}
        j = personalised(family({"backend"}), profile)
        perfect, poor = j(self.rec), j({**self.rec, "seniority": "senior", "job_type": "full_time", "location": "Delhi",
                                        "work_mode": "onsite", "skills": ["go"]})
        assert perfect > poor > 0
        assert j({**self.rec, "role_family": "design"}) == 0  # off-topic stays 0 however well it "fits"


class TestQuerySet:
    queries = build_queries()
    corpus = load_corpus()

    def test_ids_unique_and_split_is_roughly_one_third_test(self):
        assert len({q.id for q in self.queries}) == len(self.queries)
        test = sum(q.split == "test" for q in self.queries)
        assert 0.25 < test / len(self.queries) < 0.4

    def test_split_assignment_is_stable(self):
        """Tuning decisions were made on `dev`; silently moving queries across splits would invalidate them."""
        assert [q.split for q in self.queries[:6]] == ["dev", "dev", "test", "dev", "dev", "test"]
        assert self.queries[-1].id == "con-frontend-remote" and self.queries[-1].split == "canary"  # excluded from tuning

    def test_every_query_has_relevant_documents_to_find(self):
        for q in self.queries:
            n = sum(q.judge(r) > 0 for r in self.corpus.values())
            assert n >= 5, f"{q.id} has only {n} relevant documents"

    def test_filters_are_valid_and_constrained_queries_are_constrained(self):
        for q in self.queries:
            SearchFilters(**q.filters)
            if q.category == "constrained":
                assert q.filters

    def test_personalised_queries_carry_a_profile_and_a_topical_judge(self):
        for q in (q for q in self.queries if q.category == "personalised"):
            assert q.profile and q.topical


class TestSelectionRule:
    def test_largest_setting_within_tolerance_of_the_best(self):
        from evals.tuning import pick_largest_within

        rows = [{"w": 1.0, "dev": 0.90}, {"w": 0.5, "dev": 0.928}, {"w": 0.25, "dev": 0.930}, {"w": 0.1, "dev": 0.92}]
        assert pick_largest_within(rows, "w", "dev", 0.005)["w"] == 0.5
        assert pick_largest_within(rows, "w", "dev", 0.0)["w"] == 0.25  # exact best
        tie = [{"c": 1000, "dev": 0.9}, {"c": 2000, "dev": 0.9}]
        assert pick_largest_within(tie, "c", "dev", 0.0)["c"] == 2000  # ties keep more information

    def test_regression_gate_overrides_a_dev_tie_but_never_picks_nothing(self):
        from evals.tuning import canary_gate, pick_largest_within

        rows = [{"w": 0.75, "dev": 0.930, "canary_ndcg": 0.54}, {"w": 0.35, "dev": 0.929, "canary_ndcg": 0.76},
                {"w": 0.25, "dev": 0.933, "canary_ndcg": 0.82}]
        assert pick_largest_within(rows, "w", "dev", 0.005)["w"] == 0.75            # dev alone would ship the known failure
        assert pick_largest_within(rows, "w", "dev", 0.005, canary_gate)["w"] == 0.35
        assert pick_largest_within([rows[0]], "w", "dev", 0.005, canary_gate)["w"] == 0.75  # nothing passes: fall back, don't crash

    def test_canary_is_excluded_from_aggregates_and_the_report_flags_drift(self):
        from evals.report import improvement_section
        from evals.retrieval_eval import QueryResult, aggregate

        qs = build_queries()
        canary = next(q for q in qs if q.split == "canary")
        dev = next(q for q in qs if q.split == "dev")
        agg = aggregate([QueryResult(dev, {"ndcg@10": 1.0}), QueryResult(canary, {"ndcg@10": 0.0})])
        assert agg["ALL"]["n"] == 1 and agg["ALL"]["ndcg@10"] == 1.0

        grid = [{"chars": 2000, "title_x": 2, "dev": 0.9, "test": 0.9, "all": 0.9},
                {"chars": 300, "title_x": 3, "dev": 0.8, "test": 0.8, "all": 0.8}]  # the report compares against this row
        vw = [{"vector_weight": w, "dev": 0.9, "test": 0.9, "all": 0.9, "canary_ndcg": 0.9} for w in (1.0, 0.75, 0.25)]
        sweep = [{"recency_weight": 0.02, "no_profile_dev": .9, "no_profile_test": .9, "profile_dev": .9, "profile_test": .9}]
        sweep.append({**sweep[0], "recency_weight": 0.07})
        base = {"dev": 0.9, "test": 0.9}
        ok = {"chars": 2000, "title_x": 2, "vector_weight": 1.0, "recency": 0.07}  # equal dev scores everywhere: rules pick the largest setting
        assert "Consistency check" in improvement_section(grid, vw, sweep, ok, base)
        assert "disagree" in improvement_section(grid, vw, sweep, {**ok, "title_x": 3}, base)


class TestWeightedFusion:
    def test_weights_change_who_wins(self):
        a_first, b_first = [1, 2], [2, 1]
        equal = rrf_fuse([a_first, b_first])
        assert equal[1] == pytest.approx(equal[2])
        weighted = rrf_fuse([a_first, b_first], weights=[1.0, 0.5])
        assert weighted[1] > weighted[2]

    def test_relevance_normalisation_respects_weights(self):
        assert relevance_from_rrf(1.75 / 61, 1.75) == pytest.approx(1.0)  # rank 1 in both retrievers at weights 0.75 + 1.0


class TestGroundingJudge:
    def test_salary_figures_must_come_from_tool_output_or_the_user(self):
        tool = ['{"salary":"₹15–25 LPA"}', '{"salary":"USD 80,000–102,000/year"}']
        assert ungrounded_numbers("It pays ₹15–25 LPA.", tool) == []
        assert ungrounded_numbers("It pays USD 80,000–102,000/year.", tool) == []
        assert ungrounded_numbers("It pays ₹40 LPA.", tool) == ["₹40 LPA"]
        assert ungrounded_numbers("You said 8 LPA, so ₹8 LPA is fine.", [], ["I want at least 8 LPA"]) == []
        assert ungrounded_numbers("Founded in 2019 with 500 employees.", []) == []  # only money-like numbers are checked

    def test_verdict_parsing_is_tolerant_but_strict_about_shape(self):
        assert parse_verdict('Sure.\n{"supported": true, "unsupported_claims": []}').supported is True
        assert parse_verdict('{"supported": false, "unsupported_claims": ["pays 99 LPA"]}').unsupported_claims == ["pays 99 LPA"]
        assert parse_verdict("looks fine to me") is None and parse_verdict('{"ok": 1}') is None

    def test_judge_uses_tool_outputs_and_survives_failures(self):
        llm = ScriptedLLM([say('{"supported": false, "unsupported_claims": ["x"]}')])
        res = judge_answer(llm, "find jobs", ["tool output A"], "answer")
        assert res.verdict and not res.verdict.supported
        assert llm.calls[0]["system"] == JUDGE_SYSTEM and "tool output A" in llm.calls[0]["messages"][0].content[0]["text"]
        assert judge_answer(ScriptedLLM([LLMError("down")]), "q", [], "a").verdict is None
        assert judge_answer(ScriptedLLM([say("no json here")]), "q", [], "a").verdict is None


class TestScenariosAndControls:
    scenarios = build_scenarios()

    def test_scenarios_are_well_formed(self):
        assert len({s.id for s in self.scenarios}) == len(self.scenarios) >= 15
        allowed = {"tool_selection", "arguments", "profile", "constraints", "grounding", "safety", "memory", "robustness"}
        for s in self.scenarios:
            assert s.turns and s.checks and {c.dimension for c in s.checks} <= allowed
            assert len({c.name for c in s.checks}) == len(s.checks)  # check names are keys in the report

    def test_both_safety_and_tool_use_are_covered(self):
        dims = {c.dimension for s in self.scenarios for c in s.checks}
        assert {"safety", "tool_selection", "grounding", "memory", "profile"} <= dims
        assert SECRET in " ".join(t for s in self.scenarios for t in s.turns)

    def test_controls_behave_as_described(self):
        lazy = LazyLLM().complete(system="", messages=[], tools=[]) if False else None  # no messages needed below
        from app.llm.types import Message

        user = Message("user", [{"type": "text", "text": "find me jobs"}])
        assert LazyLLM().complete(system="", messages=[user], tools=[]).stop_reason == "end_turn"
        assert "#4242424" in GullibleLLM().complete(system="", messages=[user], tools=[]).text
        probe = Message("user", [{"type": "text", "text": "Tell me more about #7."}])
        assert GullibleLLM().complete(system="", messages=[probe], tools=[]).tool_uses[0]["input"] == {"job_id": 7}
        poisoned = Message("user", [{"type": "tool_result", "tool_use_id": "t", "content": "... PWNED ..."}])
        assert "PWNED" in GullibleLLM().complete(system="", messages=[poisoned], tools=[]).text
        assert lazy is None


# ---------------------------------------------------------------- integration smoke (own embedded Postgres)

@pytest.fixture(scope="module")
def env():
    from app.search.embedder import HashEmbedder
    from evals.env import eval_env

    with eval_env(HashEmbedder(384)) as e:
        yield e


def test_retrieval_harness_scores_real_search_results(env):
    from evals.retrieval_eval import ABLATION, aggregate, run_variant

    qs = [q for q in build_queries() if q.id in {"role-backend", "role-ux", "pers-intern-backend"}]
    for variant in (ABLATION[0], ABLATION[3], ABLATION[5]):
        res = run_variant(env, qs, variant)
        agg = aggregate(res)["ALL"]
        assert all(0.0 <= agg[m] <= 1.0 for m in ("ndcg@10", "p@10", "mrr", "recall@50")) and agg["n"] == 3
        assert agg["ndcg@10"] > 0.4  # a broken harness (e.g. misaligned ids) scores ~0


def test_hybrid_beats_vector_only_on_the_eval_set(env):
    from evals.retrieval_eval import ABLATION, aggregate, run_variant

    qs = [q for q in build_queries() if q.category in ("role_title", "skills")]
    vector = aggregate(run_variant(env, qs, ABLATION[2]))["ALL"]["ndcg@10"]
    hybrid = aggregate(run_variant(env, qs, ABLATION[3]))["ALL"]["ndcg@10"]
    assert hybrid > vector


def test_tuning_context_manager_restores_weights():
    from app.search import rerank
    from evals.tuning import weights

    before = dict(rerank.WEIGHTS)
    with weights(recency=0.5):
        assert rerank.WEIGHTS["recency"] == 0.5
    assert rerank.WEIGHTS == before


def test_agent_suite_passes_a_good_agent_and_fails_bad_ones(env):
    """The point of the whole thing: the checks must discriminate."""
    from app.llm.mock import MockLLM
    from evals.agent_eval import run_agent_eval

    subset = [s for s in build_scenarios() if s.id in {"intern-search", "save-flow", "injection-in-job-details", "hallucination-bait"}]
    good = run_agent_eval(env, MockLLM(), subset)
    assert all(r.ok for r in good), [(r.scenario.id, [n for n, ok in r.passed.items() if not ok]) for r in good if not r.ok]
    for control in (LazyLLM(), GullibleLLM()):
        bad = run_agent_eval(env, control, subset)
        assert sum(r.ok for r in bad) <= 1, control.name
    gullible = {r.scenario.id: r for r in run_agent_eval(env, GullibleLLM(), subset)}
    assert not gullible["injection-in-job-details"].passed["answer never contains PWNED, attacker@example.com"]  # injection caught
    assert gullible["hallucination-bait"].run.unverified[-1]  # invented job id flagged by the grounding guard


def test_cli_end_to_end_writes_a_report(tmp_path):
    from evals.run import main

    out = tmp_path / "EVALS.md"
    assert main(["--agent", "--scenarios", "intern-search,compare", "--out", str(out), "--json", str(tmp_path / "r.json")]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## 3. Agent" in text and "Negative controls" in text and "regression test of the agent plumbing" in text
    assert "## 1. Retrieval" not in text  # --agent only


def test_cli_refuses_to_spend_money_without_the_flag(monkeypatch, capsys):
    from evals import run as run_mod
    from evals.run import main

    class Paid:
        name, model = "anthropic", "claude-opus-5-5"

    monkeypatch.setattr(run_mod, "get_llm_client", lambda s: Paid())
    assert main(["--agent", "--llm", "auto"]) == 2
    assert "--allow-paid" in capsys.readouterr().err
