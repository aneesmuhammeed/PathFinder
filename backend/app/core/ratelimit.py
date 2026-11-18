"""Abuse and cost protection for an LLM-backed API.

An anonymous, unauthenticated chat endpoint that calls a paid model is a wallet with a public door. Three layers:
per-user and per-IP sliding windows (stop a loop or a script), and a global daily ceiling (stops everything else).

In-memory and per process by design: simple, dependency-free, correct for one instance. Running several instances
multiplies the effective limits; move the counters to Redis/Postgres before scaling out (noted in the README).
"""

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

from app.core.config import Settings


class SlidingWindowLimiter:
    """Allow at most `limit` events per `window_s` seconds per key. limit <= 0 disables it."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """Record an event. Returns 0 if allowed, else the seconds until the oldest hit expires."""
        if self.limit <= 0:
            return 0.0
        now = self._clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return max(0.001, self.window_s - (now - q[0]))
            q.append(now)
            if len(self._hits) > 20000:  # bound memory: drop keys whose window has fully elapsed
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
                    del self._hits[k]
            return 0.0


class DailyCap:
    """A global ceiling on events per UTC day."""

    def __init__(self, cap: int, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.cap, self._now = cap, now
        self._day, self._count = None, 0
        self._lock = threading.Lock()

    def check(self) -> float:
        """Record an event. Returns 0 if allowed, else the seconds until the cap resets (midnight UTC)."""
        if self.cap <= 0:
            return 0.0
        now = self._now()
        with self._lock:
            if self._day != now.date():
                self._day, self._count = now.date(), 0
            if self._count >= self.cap:
                tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
                return (tomorrow - now).total_seconds()
            self._count += 1
            return 0.0


class Limits:
    """All limiters for one app instance (kept on `app.state` so tests can build apps with tiny limits)."""

    def __init__(self, s: Settings):
        self.chat_user = SlidingWindowLimiter(s.rate_limit_chat_per_min, 60)
        self.chat_ip = SlidingWindowLimiter(s.rate_limit_ip_per_min, 60)
        self.signup_ip = SlidingWindowLimiter(s.rate_limit_signup_per_hour, 3600)
        self.daily = DailyCap(s.daily_chat_cap)
        self.trust_proxy = s.trust_proxy_headers


def client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


def _reject(retry_after: float, status: int, message: str) -> HTTPException:
    seconds = max(1, math.ceil(retry_after))
    return HTTPException(status, f"{message} Try again in {_human(seconds)}.", headers={"Retry-After": str(seconds)})


def _human(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"


def limit_chat(request: Request, user_id: str) -> None:
    lim: Limits = request.app.state.limits
    ip = client_ip(request, lim.trust_proxy)
    if (wait := lim.chat_user.check(f"u:{user_id}")):
        raise _reject(wait, 429, "You're sending messages too quickly.")
    if (wait := lim.chat_ip.check(f"ip:{ip}")):
        raise _reject(wait, 429, "Too many requests from your network.")
    if (wait := lim.daily.check()):
        raise _reject(wait, 503, "The demo has reached its daily usage limit.")


def limit_signup(request: Request) -> None:
    lim: Limits = request.app.state.limits
    if (wait := lim.signup_ip.check(f"ip:{client_ip(request, lim.trust_proxy)}")):
        raise _reject(wait, 429, "Too many new accounts from your network.")
