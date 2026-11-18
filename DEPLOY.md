# Deploying Pathfinder

> **Status:** these steps were written from the project's own configuration and the hosts' general behaviour. They have **not** been run end to end
> by the author (no accounts, no Docker on the build machine), and hosting dashboards change often. Where a step depends on a provider's UI, check
> its current docs. CI builds both images and smoke-tests Compose on every push, which de-risks the container side.

## What you're deploying

Three pieces: **Postgres with pgvector**, the **API** (FastAPI, one Docker image), and the **web app** (static files from `npm run build`).
Nothing else: no queue, no cache, no separate worker is required.

## Read this before you put an API key on a public URL

The chat endpoint is anonymous and calls a paid model. Defaults protect you (12 turns/min/user, 40/min/IP, **2000 turns/day total**), but:

1. Set a **spend limit in your provider's console** as the real backstop.
2. Pick a cheap model for a public demo: `LLM_MODEL=claude-sonnet-5-5` or `claude-haiku-5-5` (the default `claude-opus-5-5` is the most capable and the most expensive).
3. Tune `DAILY_CHAT_CAP` to what you're willing to spend per day. Rough arithmetic: a turn is typically 2 to 4 model calls.
4. Never commit `.env`. Put keys in the host's secret/env settings.
5. Without any key the app runs on the rule-based demo assistant (clearly labelled in the UI), which costs nothing. That is a perfectly good public demo of search, ranking, memory and the UI.

## Option A: managed Postgres + container host + static site (recommended)

### 1. Database: Neon, Supabase or any Postgres that offers pgvector
Create a database and copy its connection string. `postgres://…` / `postgresql://…` URLs work as-is. The first start runs the migrations, which execute
`CREATE EXTENSION IF NOT EXISTS vector` (the extension must be available to your database role; Neon and Supabase support it).

### 2. API: any Docker host (Render, Fly.io, Railway, Cloud Run …)
Build context is the **repo root**, Dockerfile path is `backend/Dockerfile`. The image migrates, loads the 520 demo jobs if the database is empty, then serves on `$PORT`.

| Setting | Value |
|---|---|
| Health check path | `/health` (readiness, including DB and pgvector, is `/ready`) |
| `DATABASE_URL` | from step 1 |
| `APP_ENV` | `prod` (logs warnings about anything unsafe for a public server) |
| `CORS_ORIGINS` | your web app's origin, e.g. `https://pathfinder.example.com` (comma-separated for several) |
| `TRUST_PROXY_HEADERS` | `true` (you are behind the host's proxy, so per-IP limits need `X-Forwarded-For`) |
| `DOCS_ENABLED` | `false` unless you want a public `/docs` |
| `ANTHROPIC_API_KEY` *(or OpenAI / Gemini)* | optional, see the warning above |
| `LLM_MODEL`, `DAILY_CHAT_CAP`, `RATE_LIMIT_*` | see `.env.example` |
| `LOG_JSON` | `true` |

Then check: `https://<api>/ready` returns `{"status":"ready", …}` and `https://<api>/stats` shows 520 active jobs.
Free tiers usually **sleep when idle**, so the first request after a pause is slow; the UI shows "Connecting…" meanwhile.

### 3. Web app: Vercel, Netlify, Cloudflare Pages, Render static …
```
Root directory:   frontend
Install:          npm ci
Build:            npm run build
Publish dir:      dist
Env (build time): VITE_API_URL=https://<your-api-host>      ← baked into the bundle at build time
SPA fallback:     rewrite /* to /index.html
```
Because the web app and API are on different origins, the API's `CORS_ORIGINS` must contain the web app's exact origin (scheme + host, no trailing slash).
Server-sent events work cross-origin; the API exposes `Retry-After` so rate-limit messages are readable.

### 4. Keep the job index fresh (optional)
The demo corpus is synthetic. To add live jobs, run `python -m app.cli ingest --source all` on a schedule (a cron/scheduled job in your host, every 6 hours or less often),
or run the loop as a background worker: `python -m app.cli loop --source all --interval-minutes 360`.
Remotive's terms require a visible credit and link-back (the UI shows "via Remotive") and ask for sparse polling (the code enforces a 6-hour minimum interval); don't redistribute their jobs elsewhere.
Arbeitnow asks for a link back as well.

## Option B: one VM with Docker Compose

```bash
git clone <your repo> && cd pathfinder
cp .env.example .env            # set APP_ENV=prod, keys, CORS_ORIGINS=https://your.domain
docker compose up -d --build   # UI on :8080, API on :8000
docker compose --profile live up -d    # + the live-ingestion worker
```
Put a TLS-terminating reverse proxy in front of port 8080 (Caddy is the least work):
```
your.domain {
    reverse_proxy localhost:8080
}
```
The bundled nginx already proxies `/api/` to the API with the settings the event stream needs (no buffering, no compression, long read timeout), so the browser sees a single origin and **no CORS configuration is required** in this layout.
Change the database password in `docker-compose.yml` before exposing anything; the database port is bound to localhost only.

## Embeddings in production

`EMBEDDER=hash` (default) is offline and lexical. For real semantic search use a larger instance (≥1 GB RAM) with `EMBEDDER=fastembed`,
install `backend/requirements-ml.txt` in the image (add it to the Dockerfile's `pip install`), and run `python -m app.cli reembed` once. The model (~130 MB) downloads on first use.
Then re-run `python -m evals.run`. The shipped fusion weight (`HashEmbedder.rrf_weight = 0.35`) and document recipe were tuned for the *lexical* embedder, so give the new embedder its own `rrf_weight`
(start at 1.0) and re-run the sweep in `evals/tuning.py` before trusting the numbers.

## After deploying: a 5-minute checklist

- [ ] `/ready` is green and `/stats` shows jobs with embeddings.
- [ ] Open the web app: no "Demo assistant" badge if you configured a key; the badge if you didn't.
- [ ] Send one message end to end; save a job; reload the page and confirm the memory panel is intact.
- [ ] Hit the rate limit on purpose (send 13 messages in a minute) and confirm you get a friendly "Try again in N seconds".
- [ ] Look at the logs (JSON): requests carry `request_id`; there is no message content or key in them.
- [ ] Set the provider spend limit. Re-read the warning above.

## Getting real users, honestly

A live URL is a strong signal, and "used by real people" is only worth saying if it is true. Share it with classmates and friends, ask them to try it and tell you what broke,
and keep a short list of what you changed because of their feedback. Report only numbers you can show (for example, distinct anonymous users and conversations are available from the database).
Do not inflate them: interviewers ask follow-up questions.
