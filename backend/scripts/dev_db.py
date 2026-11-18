"""Run a local PostgreSQL 16 + pgvector WITHOUT Docker (uses the `pgserver` package).

    python scripts/dev_db.py            # starts DB, prints DATABASE_URL, blocks until Ctrl+C

Data lives in backend/.pgdata so it survives restarts. Use Docker Compose for the "real" setup;
this exists so the project (and its tests) run on any laptop.
"""

import signal
import sys
import time
from pathlib import Path

import pgserver

DATA_DIR = Path(__file__).resolve().parent.parent / ".pgdata"


def main() -> None:
    server = pgserver.get_server(DATA_DIR, cleanup_mode=None)  # keep data after exit
    url = server.get_uri().replace("postgresql://", "postgresql+psycopg://")
    print("Postgres is running. In another terminal:\n")
    print(f'  PowerShell:  $env:DATABASE_URL = "{url}"')
    print(f'  bash:        export DATABASE_URL="{url}"')
    print("\nThen: alembic upgrade head   (Ctrl+C here to stop the database)")
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
