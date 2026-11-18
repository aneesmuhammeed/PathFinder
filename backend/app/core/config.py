"""Application settings, loaded from environment variables / a local .env file."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DB_URL = "postgresql+psycopg://pathfinder:pathfinder@localhost:5432/pathfinder"


class Settings(BaseSettings):
    # Docker Compose reads ./.env at the repo root; local runs start in backend/. Read both (later wins), so one
    # `.env` at the repo root configures everything and backend/.env can still override it.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"  # dev | prod
    log_level: str = "INFO"
    log_json: bool = True
    docs_enabled: bool = True  # /docs and /openapi.json; consider disabling on a public deployment

    # SQLAlchemy URL using the psycopg (v3) driver. Plain postgres:// and postgresql:// URLs, which is what hosted
    # databases (Neon, Supabase, Render...) hand out, are rewritten automatically.
    database_url: str = DEFAULT_DB_URL

    @field_validator("llm_model", "llm_effort", "anthropic_api_key", "openai_api_key", "gemini_api_key", "openai_base_url", mode="before")
    @classmethod
    def _blank_means_unset(cls, v):
        """`.env.example` ships lines like `OPENAI_BASE_URL=`; an empty variable must mean "not set", not "" (an empty
        base URL would make every OpenAI call fail)."""
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, v: str) -> str:
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v

    # --- Abuse / cost protection (in-memory, per process; use a shared store such as Redis if you scale out) ---
    rate_limit_chat_per_min: int = 12  # chat turns per anonymous user per minute (0 = unlimited)
    rate_limit_ip_per_min: int = 40  # chat turns per client IP per minute
    rate_limit_signup_per_hour: int = 30  # new anonymous users per client IP per hour
    daily_chat_cap: int = 2000  # total chat turns per UTC day across everyone: a hard ceiling on the LLM bill (0 = off)
    trust_proxy_headers: bool = False  # take the client IP from X-Forwarded-For (only behind a proxy you control)
    max_body_bytes: int = 65536

    # Bring up demo data automatically when the database is empty (the Docker image does this on start).
    bootstrap_seed: bool = True

    # Embedding dimensionality. 384 matches BAAI/bge-small-en-v1.5 (the default local model).
    embedding_dim: int = 384

    # Rolling-window sources (e.g. Arbeitnow) expire a job after it hasn't been seen for this many days.
    job_stale_after_days: int = 14

    # "hash" = deterministic, offline, keyless (lexical-ish; used for tests/CI/demo).
    # "fastembed" = real local semantic model (BAAI/bge-small-en-v1.5); needs `pip install -r requirements-ml.txt`.
    embedder: str = "hash"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    # Outbound HTTP (ingestion)
    http_timeout_s: float = 20.0
    user_agent: str = "pathfinder-ingest/0.1"
    arbeitnow_pages: int = 2

    # --- LLM ---
    # auto = first of ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY that is set, else "mock" (no key needed).
    llm_provider: str = "auto"  # auto | anthropic | openai | gemini | mock
    llm_model: str | None = None  # overrides the provider's default model
    llm_effort: str | None = None  # anthropic only: low|medium|high|xhigh|max (omit = model default)
    llm_max_tokens: int = 16000
    llm_timeout_s: float = 90.0  # per provider request (SDKs default to 10 minutes, far too long for an interactive app)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_base_url: str | None = None  # any OpenAI-compatible server (Ollama, vLLM, ...)

    # --- Agent ---
    agent_max_steps: int = 6  # model calls per user turn (guards against tool-call loops)
    agent_history_token_budget: int = 30000  # compact (summarise) the conversation above this
    max_message_chars: int = 4000

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def config_warnings(s: Settings) -> list[str]:
    """Things that are fine on a laptop and wrong on a public server. Logged at startup when APP_ENV=prod."""
    if s.app_env != "prod":
        return []
    warnings = []
    if s.database_url == DEFAULT_DB_URL or "pathfinder:pathfinder@" in s.database_url:
        warnings.append("DATABASE_URL uses the default development credentials")
    if all("localhost" in o or "127.0.0.1" in o for o in s.cors_origin_list):
        warnings.append("CORS_ORIGINS only lists localhost: the deployed web app's origin will be blocked")
    if not (s.anthropic_api_key or s.openai_api_key or s.gemini_api_key):
        warnings.append("no LLM API key is set: users will talk to the rule-based demo assistant")
    elif s.daily_chat_cap == 0 and s.rate_limit_chat_per_min == 0:
        warnings.append("a real LLM is configured with NO rate limit and NO daily cap: anyone can run up your bill")
    if s.docs_enabled:
        warnings.append("API docs (/docs) are public; set DOCS_ENABLED=false if you don't want that")
    return warnings


@lru_cache
def get_settings() -> Settings:
    return Settings()
