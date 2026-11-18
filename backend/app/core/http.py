"""Tiny resilient HTTP helper for ingestion: timeouts, bounded retries, backoff, Retry-After."""

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from app.core.config import get_settings

log = logging.getLogger("pathfinder.http_client")

RETRY_STATUS = {429, 500, 502, 503, 504}


class SourceError(RuntimeError):
    """A job source could not be fetched or returned something unusable."""


def make_client(client: httpx.Client | None = None) -> httpx.Client:
    if client is not None:
        return client
    s = get_settings()
    return httpx.Client(timeout=s.http_timeout_s, headers={"User-Agent": s.user_agent}, follow_redirects=True)


def get_json(
    url: str,
    *,
    params: dict | None = None,
    client: httpx.Client | None = None,
    retries: int = 3,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET `url` and decode JSON. Retries transport errors / 429 / 5xx with exponential backoff."""
    http = make_client(client)
    last: str = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            resp = http.get(url, params=params)
            if resp.status_code in RETRY_STATUS:
                last = f"HTTP {resp.status_code}"
                wait = backoff_s * 2 ** (attempt - 1)
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = min(float(retry_after), 30.0)
            else:
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:  # non-retryable 4xx
            raise SourceError(f"GET {url} failed: HTTP {exc.response.status_code}") from exc
        except ValueError as exc:  # bad JSON
            raise SourceError(f"GET {url} returned invalid JSON") from exc
        except httpx.TransportError as exc:
            last = f"{type(exc).__name__}: {exc}"
            wait = backoff_s * 2 ** (attempt - 1)
        if attempt < retries:
            log.warning("retrying fetch", extra={"url": url, "attempt": attempt, "reason": last})
            sleep(wait)
    raise SourceError(f"GET {url} failed after {retries} attempts ({last})")
