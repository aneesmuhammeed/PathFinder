from collections.abc import Iterable
from datetime import timedelta
from typing import Protocol

from app.ingest.schema import NormalizedJob


class JobSource(Protocol):
    name: str
    # True: each fetch returns the COMPLETE current set, so anything missing is gone (expire immediately).
    # False: fetch returns a rolling window; missing jobs may just have scrolled out (expire after a grace period).
    snapshot: bool
    # Politeness: don't fetch more often than this (None = no limit).
    min_interval: timedelta | None

    def fetch(self) -> Iterable[NormalizedJob]: ...
