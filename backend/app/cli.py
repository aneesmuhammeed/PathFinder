"""Command line: python -m app.cli <command>

    ingest [--source seed|remotive|arbeitnow|all] [--force] [--no-expire]
    loop   [--source ...] [--interval-minutes 360]
    bootstrap                   load demo jobs if the database is empty (container start-up)
    chat [--user-id UUID] [-m "one message"]   talk to the agent (mock LLM unless an API key is set)
    search "query" [--skills python,sql] [--seniority intern] [--location Gurugram] [--mode hybrid|vector|keyword]
    reembed [--source ...]      rebuild all vectors (after changing EMBEDDER)
    stats
"""

import argparse
import logging
import sys
import time

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import get_sessionmaker
from app.ingest.pipeline import reembed_all, run_ingestion
from app.ingest.sources import REGISTRY
from app.search.embedder import get_embedder

log = logging.getLogger("pathfinder.cli")


def _source_names(arg: str) -> list[str]:
    return list(REGISTRY) if arg == "all" else [arg]


def cmd_ingest(sources: list[str], force: bool, expire: bool) -> int:
    embedder = get_embedder()
    failed = 0
    for name in sources:
        # One session per source so a failure in one can't poison the next.
        with get_sessionmaker()() as session:
            run = run_ingestion(session, REGISTRY[name](), embedder, force=force, expire=expire)
        if run is None:
            print(f"{name:10} skipped (ran recently; use --force)")
            continue
        print(
            f"{name:10} {run.status:8} fetched={run.fetched} inserted={run.inserted} updated={run.updated} "
            f"unchanged={run.unchanged} embedded={run.embedded} expired={run.expired}"
            + (f"  [{run.error}]" if run.error else "")
        )
        failed += run.status == "failed"
    return 1 if failed else 0


def cmd_loop(sources: list[str], interval_minutes: float) -> int:
    log.info("starting ingestion loop", extra={"sources": sources, "interval_minutes": interval_minutes})
    while True:
        cmd_ingest(sources, force=False, expire=True)  # per-source min_interval keeps us polite
        time.sleep(interval_minutes * 60)


def cmd_bootstrap() -> int:
    """Container start-up: if the jobs table is empty, load the demo corpus so a fresh deployment isn't blank.
    Idempotent and safe to run on every start (does nothing when data exists or BOOTSTRAP_SEED=false)."""
    from sqlalchemy import func, select

    from app.db.models import Job

    if not get_settings().bootstrap_seed:
        print("bootstrap: BOOTSTRAP_SEED=false, skipping")
        return 0
    with get_sessionmaker()() as session:
        if session.scalar(select(func.count()).select_from(Job)):
            print("bootstrap: jobs already present, nothing to do")
            return 0
        run = run_ingestion(session, REGISTRY["seed"](), get_embedder(), force=True)
    print(f"bootstrap: loaded {run.inserted} demo jobs ({run.status})")
    return 0 if run.status == "success" else 1


def cmd_reembed(source: str | None) -> int:
    embedder = get_embedder()
    with get_sessionmaker()() as session:
        n = reembed_all(session, embedder, source)
    print(f"re-embedded {n} jobs with embedder={embedder.name}")
    return 0


def cmd_search(a: argparse.Namespace) -> int:
    from app.search.schema import ProfileSignals, SearchFilters, SearchRequest
    from app.search.service import InvalidSearch, search_jobs

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else []  # noqa: E731
    profile = ProfileSignals(
        skills=split(a.skills), seniority=a.seniority, job_type=a.job_type, work_mode=a.work_mode,
        preferred_locations=split(a.location), target_roles=split(a.roles),
    )
    req = SearchRequest(
        query=a.query, limit=a.limit, mode=a.mode, rerank=not a.no_rerank, profile=profile,
        filters=SearchFilters(locations=split(a.location) or None, include_remote=bool(a.location)),
    )
    try:
        with get_sessionmaker()() as session:
            res = search_jobs(session, get_embedder(), req)
    except InvalidSearch as exc:
        print(f"error: {exc}")
        return 2
    exp = f"  (+{', '.join(res.expansions)})" if res.expansions else ""
    print(f"query: {res.query_used!r}{exp}  mode={res.mode} personalised={res.personalised} {res.took_ms}ms\n")
    for i, r in enumerate(res.results, 1):
        j = r.job
        print(f"{i:2}. {r.score:.2f}  {j.title} @ {j.company}  [{j.seniority}, {j.location}, {j.work_mode}]  #{j.id}")
        for why in r.reasons[:3]:
            print(f"        + {why}")
        for why in r.caveats[:2]:
            print(f"        - {why}")
    return 0


def cmd_chat(user_id: str | None, message: str | None) -> int:
    """Talk to the agent in the terminal. Uses LLM_PROVIDER/keys from the environment (mock if none)."""
    import uuid

    from app.agent.loop import run_turn
    from app.llm import get_llm_client
    from app.search.embedder import get_embedder as _embedder
    from app.services import users as users_svc

    llm, embedder = get_llm_client(), _embedder()
    with get_sessionmaker()() as session:
        if user_id:
            uid = uuid.UUID(user_id)
        else:
            uid = users_svc.create_user(session).id
            session.commit()
            print(f"(new anonymous user {uid}; pass --user-id to continue as them)")
        print(f"provider: {llm.name} ({llm.model})" + ("  [rule-based mock: no API key set]" if llm.name == "mock" else ""))
        conversation_id = None
        pending = [message] if message else None
        while True:
            try:
                text = pending.pop(0) if pending else (None if message else input("\nyou> ").strip())
            except (EOFError, KeyboardInterrupt):
                return 0
            if not text:
                if message:
                    return 0
                continue
            if text.lower() in {"exit", "quit"}:
                return 0
            for ev in run_turn(session, llm, embedder, user_id=uid, text=text, conversation_id=conversation_id):
                if ev["type"] == "conversation":
                    conversation_id = uuid.UUID(ev["conversation_id"])
                elif ev["type"] == "tool_call":
                    print(f"  · {ev['name']}({', '.join(f'{k}={v!r}' for k, v in ev['input'].items())[:100]})")
                elif ev["type"] == "tool_result" and not ev["ok"]:
                    print(f"  · ! {ev['summary']}")
                elif ev["type"] == "message":
                    print("\n" + ev["text"])
                elif ev["type"] == "error":
                    print(f"\n[error] {ev['message']}")
            if message:
                return 0


def cmd_stats() -> int:
    from sqlalchemy import text

    with get_sessionmaker()() as s:
        rows = s.execute(text(
            "SELECT source, count(*) FILTER (WHERE expired_at IS NULL) AS active, "
            "count(*) FILTER (WHERE expired_at IS NOT NULL) AS expired, "
            "count(*) FILTER (WHERE embedding IS NOT NULL) AS embedded FROM jobs GROUP BY source ORDER BY source"
        )).all()
    print(f"{'source':10} {'active':>7} {'expired':>8} {'embedded':>9}")
    for r in rows:
        print(f"{r.source:10} {r.active:7} {r.expired:8} {r.embedded:9}")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    p = argparse.ArgumentParser(prog="pathfinder")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "loop"):
        sp = sub.add_parser(name)
        sp.add_argument("--source", default="all", choices=[*REGISTRY, "all"])
        if name == "ingest":
            sp.add_argument("--force", action="store_true", help="ignore per-source rate limits")
            sp.add_argument("--no-expire", action="store_true")
        else:
            sp.add_argument("--interval-minutes", type=float, default=360)
    sub.add_parser("stats")
    sub.add_parser("bootstrap", help="load the demo jobs if the database is empty (used on container start)")
    sp = sub.add_parser("chat", help="talk to the agent in the terminal")
    sp.add_argument("--user-id", help="continue as an existing user (keeps their saved profile)")
    sp.add_argument("--message", "-m", help="send one message and exit")
    sp = sub.add_parser("search", help="try a search from the terminal")
    sp.add_argument("query", nargs="?", default="")
    sp.add_argument("--skills", default="", help="comma-separated, e.g. python,sql")
    sp.add_argument("--roles", default="", help="target roles (used when no query is given)")
    sp.add_argument("--seniority", choices=["intern", "junior", "mid", "senior", "lead"])
    sp.add_argument("--job-type", dest="job_type", choices=["full_time", "internship", "contract", "part_time"])
    sp.add_argument("--work-mode", dest="work_mode", choices=["remote", "hybrid", "onsite"])
    sp.add_argument("--location", default="", help="comma-separated; remote jobs are kept too")
    sp.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    sp.add_argument("--no-rerank", action="store_true")
    sp.add_argument("--limit", type=int, default=8)
    sp = sub.add_parser("reembed", help="rebuild all vectors (after changing EMBEDDER)")
    sp.add_argument("--source", choices=list(REGISTRY), default=None)
    a = p.parse_args(argv)

    if a.cmd == "ingest":
        return cmd_ingest(_source_names(a.source), a.force, not a.no_expire)
    if a.cmd == "loop":
        return cmd_loop(_source_names(a.source), a.interval_minutes)
    if a.cmd == "reembed":
        return cmd_reembed(a.source)
    if a.cmd == "search":
        return cmd_search(a)
    if a.cmd == "chat":
        return cmd_chat(a.user_id, a.message)
    if a.cmd == "bootstrap":
        return cmd_bootstrap()
    return cmd_stats()


if __name__ == "__main__":
    sys.exit(main())
