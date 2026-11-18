def test_health_is_up_without_db(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.headers["X-Request-ID"]


def test_health_reports_which_llm_backs_the_app(client, monkeypatch):
    from app.core.config import get_settings

    get_settings.cache_clear()
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert client.get("/health").json()["llm"] == "mock"  # a UI can show a "demo assistant" banner honestly
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    get_settings.cache_clear()
    assert client.get("/health").json()["llm"] == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    get_settings.cache_clear()


def test_request_id_is_propagated(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_ready_checks_db_pgvector_and_migration(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ready",
        "checks": {"database": True, "pgvector": True, "migrated": True},
    }
