from app.ingest.sources.arbeitnow import ArbeitnowSource
from app.ingest.sources.base import JobSource
from app.ingest.sources.remotive import RemotiveSource
from app.ingest.sources.seed import SeedSource

REGISTRY: dict[str, type] = {"seed": SeedSource, "remotive": RemotiveSource, "arbeitnow": ArbeitnowSource}

__all__ = ["REGISTRY", "JobSource", "SeedSource", "RemotiveSource", "ArbeitnowSource"]
