"""Abuse protection, deployment config and request hygiene."""

import logging
from contextlib import nullcontext
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.api.deps import get_embedder_dep, get_llm_dep, get_session_factory
from app.core.config import DEFAULT_DB_URL, Settings, config_warnings
from app.core.ratelimit import DailyCap, SlidingWindowLimiter
from app.db.session import get_db
from app.llm.mock import MockLLM
from app.services import users as svc
from tests.helpers import EMB


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------- limiter logic

class TestSlidingWindow:
    def test_allows_up_to_the_limit_then_reports_how_long_to_wait(self):
        clock = Clock()
        lim = SlidingWindowLimiter(3, 60, clock)
        assert [lim.check("a") for _ in range(3)] == [0, 0, 0]
        clock.t += 10
        wait = lim.check("a")
        assert wait == pytest.approx(50.0)  # until the oldest hit (t=1000) leaves the window

    def test_recovers_as_the_window_slides(self):
        clock = Clock()
        lim = SlidingWindowLimiter(2, 60, clock)
        lim.check("a"); clock.t += 30; lim.check("a")
        assert lim.check("a") > 0
        clock.t += 31  # first hit expired, second still counts
        assert lim.check("a") == 0 and lim.check("a") > 0

    def test_keys_are_independent_and_rejections_do_not_extend_the_block(self):
        clock = Clock()
        lim = SlidingWindowLimiter(1, 60, clock)
        assert lim.check("a") == 0 and lim.check("b") == 0
        for _ in range(100):  # hammering while blocked must not push the unblock time out
            clock.t += 0.1
            lim.check("a")
        clock.t = 1060.0
        assert lim.check("a") == 0

    def test_zero_disables(self):
        lim = SlidingWindowLimiter(0, 60)
        assert all(lim.check("a") == 0 for _ in range(1000))

    def test_memory_is_bounded(self):
        clock = Clock()
        lim = SlidingWindowLimiter(1, 1, clock)
        for i in range(20001):
            lim.check(f"k{i}")
        clock.t += 5
        lim.check("trigger")  # next insertion purges expired keys
        assert len(lim._hits) < 100


class TestDailyCap:
    def test_counts_resets_at_midnight_utc_and_reports_time_left(self):
        now = {"v": datetime(2026, 11, 18, 23, 0, tzinfo=timezone.utc)}
        cap = DailyCap(2, lambda: now["v"])
        assert cap.check() == 0 and cap.check() == 0
        assert cap.check() == pytest.approx(3600)  # one hour to midnight
        now["v"] = datetime(2026, 11, 19, 0, 0, 1, tzinfo=timezone.utc)
        assert cap.check() == 0  # new day, fresh budget

    def test_zero_means_unlimited(self):
        cap = DailyCap(0)
        assert all(cap.check() == 0 for _ in range(500))


# ---------------------------------------------------------------- HTTP behaviour

def build(db, monkeypatch, **settings) -> TestClient:
    monkeypatch.setattr(main_mod, "get_settings", lambda: Settings(**settings))
    app = main_mod.create_app()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_session_factory] = lambda: (lambda: nullcontext(db))
    app.dependency_overrides[get_llm_dep] = lambda: MockLLM()
    app.dependency_overrides[get_embedder_dep] = lambda: EMB
    return TestClient(app)


def chat(api, user, text="hello", **headers):
    return api.post("/chat", json={"user_id": str(user.id), "message": text, "stream": False}, headers=headers)


def test_chat_is_rate_limited_per_user_with_a_helpful_retry_hint(db, monkeypatch):
    api = build(db, monkeypatch, rate_limit_chat_per_min=2, rate_limit_ip_per_min=0, daily_chat_cap=0)
    alice, bob = svc.create_user(db), svc.create_user(db)
    assert [chat(api, alice).status_code for _ in range(2)] == [200, 200]
    blocked = chat(api, alice)
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60
    assert "too quickly" in blocked.json()["detail"] and "Try again in" in blocked.json()["detail"]
    assert chat(api, bob).status_code == 200  # someone else is unaffected


def test_chat_is_rate_limited_per_ip(db, monkeypatch):
    api = build(db, monkeypatch, rate_limit_chat_per_min=0, rate_limit_ip_per_min=2, daily_chat_cap=0)
    users = [svc.create_user(db) for _ in range(3)]
    assert [chat(api, u).status_code for u in users] == [200, 200, 429]  # one network, three accounts


def test_forwarded_for_is_ignored_unless_the_proxy_is_trusted(db, monkeypatch):
    spoof = lambda i: {"X-Forwarded-For": f"10.0.0.{i}"}  # noqa: E731
    untrusted = build(db, monkeypatch, rate_limit_chat_per_min=0, rate_limit_ip_per_min=1, daily_chat_cap=0)
    users = [svc.create_user(db) for _ in range(2)]
    assert [chat(untrusted, u, **spoof(i)).status_code for i, u in enumerate(users)] == [200, 429]  # spoofing doesn't help

    trusted = build(db, monkeypatch, rate_limit_chat_per_min=0, rate_limit_ip_per_min=1, daily_chat_cap=0, trust_proxy_headers=True)
    assert [chat(trusted, u, **spoof(i)).status_code for i, u in enumerate(users)] == [200, 200]  # distinct clients behind the proxy


def test_signups_are_limited_per_ip(db, monkeypatch):
    api = build(db, monkeypatch, rate_limit_signup_per_hour=2)
    assert [api.post("/users", json={}).status_code for _ in range(3)] == [201, 201, 429]


def test_daily_cap_is_a_hard_ceiling_for_everyone(db, monkeypatch):
    api = build(db, monkeypatch, rate_limit_chat_per_min=0, rate_limit_ip_per_min=0, daily_chat_cap=2)
    users = [svc.create_user(db) for _ in range(3)]
    codes = [chat(api, u) for u in users]
    assert [r.status_code for r in codes] == [200, 200, 503]
    assert "daily usage limit" in codes[2].json()["detail"] and int(codes[2].headers["Retry-After"]) > 0


def test_invalid_requests_do_not_burn_the_budget(db, monkeypatch):
    api = build(db, monkeypatch, rate_limit_chat_per_min=1, rate_limit_ip_per_min=1, daily_chat_cap=1)
    user = svc.create_user(db)
    import uuid

    assert api.post("/chat", json={"user_id": str(uuid.uuid4()), "message": "hi", "stream": False}).status_code == 404
    assert api.post("/chat", json={"user_id": str(user.id), "message": "   ", "stream": False}).status_code == 422
    assert chat(api, user).status_code == 200  # the one real turn still has its allowance


def test_rate_limited_streaming_request_is_a_clean_http_error_not_a_broken_stream(db, monkeypatch):
    api = build(db, monkeypatch, rate_limit_chat_per_min=1, rate_limit_ip_per_min=0, daily_chat_cap=0)
    user = svc.create_user(db)
    api.post("/chat", json={"user_id": str(user.id), "message": "hi"})
    r = api.post("/chat", json={"user_id": str(user.id), "message": "hi"})
    assert r.status_code == 429 and r.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------- request hygiene

def test_request_id_is_sanitised_and_security_headers_are_set(db, monkeypatch):
    api = build(db, monkeypatch)
    ok = api.get("/health", headers={"X-Request-ID": "abc-123_x.y"})
    assert ok.headers["X-Request-ID"] == "abc-123_x.y"
    for hostile in ["a b", "x" * 200, "id<script>", "line\tbreak"]:
        r = api.get("/health", headers={"X-Request-ID": hostile})
        assert r.headers["X-Request-ID"] != hostile and len(r.headers["X-Request-ID"]) == 12  # replaced, not echoed
    assert ok.headers["X-Content-Type-Options"] == "nosniff" and ok.headers["X-Frame-Options"] == "DENY"
    assert ok.headers["Referrer-Policy"] == "no-referrer"


def test_oversized_bodies_are_rejected_before_parsing(db, monkeypatch):
    api = build(db, monkeypatch, max_body_bytes=1000)
    r = api.post("/users", content=b"x" * 2000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and "too large" in r.json()["detail"]


def test_docs_can_be_switched_off(db, monkeypatch):
    assert build(db, monkeypatch, docs_enabled=True).get("/docs").status_code == 200
    off = build(db, monkeypatch, docs_enabled=False)
    assert off.get("/docs").status_code == 404 and off.get("/openapi.json").status_code == 404


# ---------------------------------------------------------------- configuration

class TestConfig:
    @pytest.mark.parametrize("given", ["postgres://u:p@host/db?sslmode=require", "postgresql://u:p@host/db"])
    def test_hosted_database_urls_are_rewritten_for_the_psycopg_driver(self, given):
        s = Settings(database_url=given)
        assert s.database_url.startswith("postgresql+psycopg://u:p@host/db")
        assert ("sslmode=require" in s.database_url) == ("sslmode" in given)  # query string preserved

    def test_blank_values_from_env_example_mean_unset(self):
        """Regression: `OPENAI_BASE_URL=` in a copied .env.example became "" and broke every OpenAI call."""
        s = Settings(openai_base_url="", anthropic_api_key="  ", gemini_api_key="", llm_model="", llm_effort="")
        assert (s.openai_base_url, s.anthropic_api_key, s.gemini_api_key, s.llm_model, s.llm_effort) == (None,) * 5
        from app.llm.factory import resolve_provider

        assert resolve_provider(s) == "mock"  # blank keys don't count as keys

    def test_a_dotenv_file_copied_from_the_example_behaves_like_no_config(self, tmp_path, monkeypatch):
        import re
        from pathlib import Path

        example = Path(__file__).resolve().parents[2] / ".env.example"
        env = tmp_path / ".env"
        env.write_text(re.sub(r"^DATABASE_URL=.*$", "", example.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL", "LLM_EFFORT", "DATABASE_URL"):
            monkeypatch.delenv(k, raising=False)
        s = Settings(_env_file=str(env))
        assert s.openai_base_url is None and s.llm_model is None
        from app.llm.factory import resolve_provider

        assert resolve_provider(s) == "mock"

    def test_dotenv_is_found_at_the_repo_root_and_in_backend(self, tmp_path, monkeypatch):
        """One `.env` at the repo root must configure local runs too (Docker Compose reads that same file)."""
        root, backend = tmp_path, tmp_path / "backend"
        backend.mkdir()
        (root / ".env").write_text("DAILY_CHAT_CAP=7\nRATE_LIMIT_CHAT_PER_MIN=3\n", encoding="utf-8")
        (backend / ".env").write_text("RATE_LIMIT_CHAT_PER_MIN=99\n", encoding="utf-8")  # backend/.env overrides the root one
        for k in ("DAILY_CHAT_CAP", "RATE_LIMIT_CHAT_PER_MIN"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.chdir(backend)
        s = Settings()
        assert s.daily_chat_cap == 7 and s.rate_limit_chat_per_min == 99

    def test_an_explicit_driver_is_left_alone(self):
        assert Settings(database_url="postgresql+psycopg://a:b@h/d").database_url == "postgresql+psycopg://a:b@h/d"

    def test_production_misconfiguration_is_called_out(self):
        assert config_warnings(Settings(app_env="dev")) == []  # nothing to say about a laptop
        bad = " | ".join(config_warnings(Settings(app_env="prod", database_url=DEFAULT_DB_URL, anthropic_api_key="k",
                                                  daily_chat_cap=0, rate_limit_chat_per_min=0)))
        for needle in ("default development credentials", "CORS_ORIGINS", "NO rate limit", "/docs"):
            assert needle in bad
        good = config_warnings(Settings(app_env="prod", database_url="postgresql+psycopg://u:s3cret@h/d", cors_origins="https://app.example.com",
                                        anthropic_api_key="k", docs_enabled=False))
        assert good == []

    def test_prod_with_no_key_warns_about_the_demo_assistant(self):
        assert any("rule-based" in w for w in config_warnings(Settings(app_env="prod", docs_enabled=False)))

    def test_warnings_are_logged_at_startup(self, monkeypatch):
        seen: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record):
                seen.append(record.getMessage())

        logger = logging.getLogger("pathfinder.http")
        handler = Capture(level=logging.WARNING)
        logger.addHandler(handler)
        try:
            monkeypatch.setattr(main_mod, "get_settings", lambda: Settings(app_env="prod"))
            main_mod.create_app()
        finally:
            logger.removeHandler(handler)
        assert any("configuration warning" in m for m in seen)

    def test_sensible_defaults(self):
        fields = Settings.model_fields  # (not Settings(): the test process overrides DATABASE_URL via the environment)
        assert fields["database_url"].default == DEFAULT_DB_URL
        assert fields["llm_timeout_s"].default == 90 and fields["daily_chat_cap"].default > 0
        assert fields["trust_proxy_headers"].default is False and fields["docs_enabled"].default is True


def test_llm_clients_get_a_bounded_timeout():
    from app.llm.factory import get_llm_client

    a = get_llm_client(Settings(anthropic_api_key="k", llm_timeout_s=45))
    assert a._client.timeout == 45
    o = get_llm_client(Settings(openai_api_key="k", llm_timeout_s=45))
    assert o._client.timeout == 45


# ---------------------------------------------------------------- container bootstrap

def test_bootstrap_loads_demo_data_once(db, monkeypatch, capsys):
    from sqlalchemy import func, select

    from app import cli
    from app.db.models import Job

    if db.scalar(select(func.count()).select_from(Job)):
        pytest.skip("test database already has jobs")
    monkeypatch.setattr(cli, "get_sessionmaker", lambda: (lambda: nullcontext(db)))
    assert cli.cmd_bootstrap() == 0
    first = db.scalar(select(func.count()).select_from(Job))
    assert first >= 500 and "loaded" in capsys.readouterr().out
    assert cli.cmd_bootstrap() == 0  # second start: nothing to do
    assert db.scalar(select(func.count()).select_from(Job)) == first and "already present" in capsys.readouterr().out


def test_bootstrap_can_be_disabled(monkeypatch, capsys):
    from app import cli

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(bootstrap_seed=False))
    assert cli.cmd_bootstrap() == 0 and "skipping" in capsys.readouterr().out
