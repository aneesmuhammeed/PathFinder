# Build from the repo root:  docker build -f backend/Dockerfile -t pathfinder-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
WORKDIR /srv

COPY backend/requirements.txt backend/requirements-llm.txt backend/
RUN pip install -r backend/requirements-llm.txt

COPY backend/ backend/
COPY data/ data/

RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

WORKDIR /srv/backend
EXPOSE 8000
# $PORT is injected by hosts like Render/Fly/Railway; locally it stays 8000.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT','8000'))" || exit 1

# 1) migrate  2) load demo jobs if the database is empty (BOOTSTRAP_SEED=false to skip)  3) serve.
# Fine for a single instance; split migrations into a release step when scaling out.
CMD ["sh", "-c", "alembic upgrade head && python -m app.cli bootstrap && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
