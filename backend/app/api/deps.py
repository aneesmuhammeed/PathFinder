from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import lru_cache

from sqlalchemy.orm import Session

from app.db.session import get_sessionmaker
from app.llm import LLMClient, get_llm_client
from app.search.embedder import Embedder, get_embedder


@lru_cache
def get_embedder_dep() -> Embedder:
    """Process-wide embedder (loading a model is expensive). Overridable in tests via dependency_overrides."""
    return get_embedder()


@lru_cache
def get_llm_dep() -> LLMClient:
    return get_llm_client()


SessionFactory = Callable[[], AbstractContextManager[Session]]


def get_session_factory() -> SessionFactory:
    """Streaming responses outlive the request-scoped `get_db` session, so /chat opens its own."""
    return lambda: get_sessionmaker()()
