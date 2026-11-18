import logging

from app.core.config import Settings, get_settings
from app.llm.types import LLMClient

log = logging.getLogger("pathfinder.llm")


def resolve_provider(s: Settings) -> str:
    if s.llm_provider != "auto":
        return s.llm_provider
    if s.anthropic_api_key:
        return "anthropic"
    if s.openai_api_key:
        return "openai"
    if s.gemini_api_key:
        return "gemini"
    return "mock"


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    provider = resolve_provider(s)
    if provider == "anthropic":
        from app.llm.anthropic_client import AnthropicClient

        client: LLMClient = AnthropicClient(s.llm_model, s.anthropic_api_key, s.llm_effort, timeout=s.llm_timeout_s)
    elif provider == "openai":
        from app.llm.openai_client import OpenAICompatClient

        client = OpenAICompatClient(name="openai", model=s.llm_model, api_key=s.openai_api_key, base_url=s.openai_base_url or None,
                                    timeout=s.llm_timeout_s)
    elif provider == "gemini":
        from app.llm.openai_client import GEMINI_BASE_URL, GEMINI_DEFAULT_MODEL, OpenAICompatClient

        client = OpenAICompatClient(name="gemini", model=s.llm_model or GEMINI_DEFAULT_MODEL,
                                    api_key=s.gemini_api_key, base_url=GEMINI_BASE_URL, timeout=s.llm_timeout_s)
    elif provider == "mock":
        from app.llm.mock import MockLLM

        client = MockLLM()
    else:
        raise ValueError(f"unknown LLM_PROVIDER={provider!r} (auto|anthropic|openai|gemini|mock)")
    log.info("llm provider selected", extra={"provider": client.name, "model": client.model})
    return client
