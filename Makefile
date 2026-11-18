# Convenience targets (on Windows without make, run the commands in README.md directly).
PY ?= backend/.venv/bin/python
ifeq ($(OS),Windows_NT)
PY = backend/.venv/Scripts/python
endif

.PHONY: setup test db dev web web-test migrate run ingest eval eval-llm chat stats seed-data up down

setup:            ## create venv + install deps
	python -m venv backend/.venv
	$(PY) -m pip install -r backend/requirements-dev.txt

test:             ## run tests (embedded Postgres, no Docker needed)
	cd backend && ../$(PY) -m pytest -q

db:               ## Postgres+pgvector without Docker (blocks; prints DATABASE_URL)
	cd backend && ../$(PY) scripts/dev_db.py

migrate:          ## apply DB migrations
	cd backend && ../$(PY) -m alembic upgrade head

run:              ## API with autoreload
	cd backend && ../$(PY) -m uvicorn app.main:app --reload

ingest:           ## load seed + live sources (needs DATABASE_URL; live sources are rate-limited)
	cd backend && ../$(PY) -m app.cli ingest

dev:              ## whole backend, no Docker (embedded Postgres + seed data); then run `make web` in another terminal
	cd backend && ../$(PY) scripts/dev_stack.py

web:              ## the chat UI on http://localhost:5173 (needs the backend running)
	cd frontend && npm install && npm run dev

web-test:         ## frontend type-check + unit/integration tests
	cd frontend && npm run typecheck && npm test

eval:             ## free evals (retrieval ablations + agent regression suite on the mock); writes EVALS.md
	cd backend && ../$(PY) -m evals.run

eval-llm:         ## agent evals against a REAL model (costs money: needs an API key and --allow-paid)
	cd backend && ../$(PY) -m evals.run --agent --llm auto --allow-paid --out ../EVALS-agent.md

chat:             ## talk to the agent in the terminal (mock LLM unless an API key is set)
	cd backend && ../$(PY) -m app.cli chat

stats:            ## index counts per source
	cd backend && ../$(PY) -m app.cli stats

seed-data:        ## regenerate data/seed_jobs.jsonl (deterministic)
	cd backend && ../$(PY) scripts/generate_seed.py

up:               ## full stack with Docker
	docker compose up --build

down:
	docker compose down
