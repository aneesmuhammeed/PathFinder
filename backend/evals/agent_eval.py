"""Run agent scenarios end to end (real agent loop, real tools, real database) with a given LLM and score the checks."""

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select

from app.agent.loop import AgentConfig, run_turn
from app.db.models import Job
from app.db.models import Message as DBMessage
from app.ingest.pipeline import run_ingestion
from app.ingest.schema import NormalizedJob
from app.llm.types import LLMClient
from app.services import users as users_svc
from evals.agent_set import INJECTION_PAYLOAD, Scenario, ScenarioRun, ToolCallRecord
from evals.env import EvalEnv
from evals.judge import JudgeResult, judge_answer, ungrounded_numbers


class _InjectedSource:
    name, snapshot, min_interval = "eval", True, None

    def fetch(self):
        return [NormalizedJob(
            source="eval", external_id="injected-1", company="Hexa Trap Labs",
            title="Python Backend Intern (FastAPI)", location="Bengaluru", country="India", work_mode="hybrid",
            job_type="internship", seniority="intern", skills=["python", "fastapi", "sql"],
            description="Join our backend team and build APIs with Python and FastAPI. " + INJECTION_PAYLOAD)]


def add_injected_job(env: EvalEnv) -> int:
    run_ingestion(env.session, _InjectedSource(), env.embedder, now=env.now, force=True)
    return env.session.scalar(select(Job.id).where(Job.source == "eval", Job.external_id == "injected-1"))


@dataclass
class ScenarioResult:
    scenario: Scenario
    run: ScenarioRun
    passed: dict[str, bool]  # check name -> passed
    dims: dict[str, str]  # check name -> dimension
    ungrounded_numbers: list[str] = field(default_factory=list)
    judge: JudgeResult | None = None

    @property
    def ok(self) -> bool:
        return all(self.passed.values())


def _fill(text: str, state: dict) -> str | None:
    try:
        return text.format(**{k: state.get(k) for k in ("first_job_id", "second_job_id", "injected_id")}) if "{" in text else text
    except (KeyError, IndexError):
        return None


def _collect_tool_calls(env: EvalEnv, conv_id: uuid.UUID, turn_starts: list[int]) -> list[ToolCallRecord]:
    msgs = list(env.session.scalars(select(DBMessage).where(DBMessage.conversation_id == conv_id).order_by(DBMessage.id)))
    pending: dict[str, ToolCallRecord] = {}
    calls: list[ToolCallRecord] = []
    for m in msgs:
        turn = max(i for i, start in enumerate(turn_starts) if start <= m.id) if turn_starts else 0
        for b in m.content:
            if b.get("type") == "tool_use":
                pending[b["id"]] = ToolCallRecord(turn, b["name"], b["input"], "", False)
                calls.append(pending[b["id"]])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in pending:
                rec = pending[b["tool_use_id"]]
                rec.output, rec.is_error = b["content"], bool(b.get("is_error"))
    return calls


def run_scenario(env: EvalEnv, llm: LLMClient, sc: Scenario, injected_id: int, judge_llm: LLMClient | None = None) -> ScenarioResult:
    user = users_svc.create_user(env.session)
    env.session.commit()
    if sc.preset_profile:
        users_svc.update_profile(env.session, user.id, dict(sc.preset_profile))
        env.session.commit()
    cfg = AgentConfig(history_token_budget=sc.history_token_budget or AgentConfig().history_token_budget)
    run = ScenarioRun(state={"injected_id": injected_id})
    conv_id: uuid.UUID | None = None
    turn_starts: list[int] = []
    t0 = time.perf_counter()

    for raw in sc.turns:
        text = _fill(raw, run.state)
        if text is None:  # a placeholder had nothing to fill it (e.g. the first search returned no jobs)
            run.errors.append(f"could not build turn: {raw!r}")
            break
        run.turn_texts.append(text)
        last_id = env.session.scalar(select(DBMessage.id).order_by(DBMessage.id.desc()).limit(1)) or 0
        turn_starts.append(last_id + 1)
        final = None
        try:
            for ev in run_turn(env.session, llm, env.embedder, user_id=user.id, text=text, conversation_id=conv_id, config=cfg, now=env.now):
                if ev["type"] == "conversation":
                    conv_id = uuid.UUID(ev["conversation_id"])
                elif ev["type"] == "message":
                    final = ev
                elif ev["type"] == "error":
                    run.errors.append(ev["message"])
        except Exception as exc:  # noqa: BLE001 - a crash is a failed scenario, not a failed eval run
            env.session.rollback()
            run.errors.append(f"crash: {type(exc).__name__}: {exc}")
            break
        if final:
            run.answers.append(final["text"])
            run.cards.append(final["jobs"])
            run.unverified.append(final["grounding"]["unverified"])
            run.steps += final["steps"]
            run.input_tokens += final["usage"]["input_tokens"]
            run.output_tokens += final["usage"]["output_tokens"]
            ids = [j["id"] for j in final["jobs"]]
            for key, idx in (("first_job_id", 0), ("second_job_id", 1)):
                if key not in run.state and len(ids) > idx:  # first one ever shown stays "the first job"
                    run.state[key] = ids[idx]
    run.seconds = time.perf_counter() - t0

    run.calls = _collect_tool_calls(env, conv_id, turn_starts) if conv_id else []
    prof = users_svc.get_profile(env.session, user.id)
    run.profile = {**users_svc.profile_signals(prof).model_dump(), "summary": prof.summary, "facts": prof.facts}
    run.saved_ids = {j.id for j in users_svc.feedback_jobs(env.session, user.id, "saved")}

    passed, dims = {}, {}
    for c in sc.checks:
        try:
            passed[c.name] = bool(c.fn(run))
        except Exception:  # noqa: BLE001 - e.g. no answer at all
            passed[c.name] = False
        dims[c.name] = c.dimension
    outputs = [c.output for c in run.calls if not c.is_error]
    bad_numbers = [n for a in run.answers for n in ungrounded_numbers(a, outputs, run.turn_texts)]
    result = ScenarioResult(sc, run, passed, dims, bad_numbers)
    if judge_llm and run.answers:
        result.judge = judge_answer(judge_llm, " / ".join(run.turn_texts), outputs, run.answers[-1])
    return result


def run_agent_eval(env: EvalEnv, llm: LLMClient, scenarios: list[Scenario], judge_llm: LLMClient | None = None) -> list[ScenarioResult]:
    injected_id = add_injected_job(env)
    return [run_scenario(env, llm, sc, injected_id, judge_llm) for sc in scenarios]


def by_dimension(results: list[ScenarioResult]) -> dict[str, tuple[int, int]]:
    """dimension -> (checks passed, checks total)."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        for name, ok in r.passed.items():
            out[r.dims[name]][0] += ok
            out[r.dims[name]][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}
