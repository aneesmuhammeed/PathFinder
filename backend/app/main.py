import logging
import re
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import chat, health, search, stats, users
from app.core.config import config_warnings, get_settings
from app.core.logging import configure_logging, request_id_ctx
from app.core.ratelimit import Limits

log = logging.getLogger("pathfinder.http")

# Caller-supplied request ids end up in logs and response headers: accept only a short, boring charset.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    for w in config_warnings(settings):
        log.warning("configuration warning: %s", w)

    app = FastAPI(
        title="Pathfinder",
        version=__version__,
        description="Conversational AI job-search agent",
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.limits = Limits(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        rid = supplied if _SAFE_REQUEST_ID.match(supplied) else uuid.uuid4().hex[:12]
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
                response = JSONResponse({"detail": "Request body too large."}, status_code=413)
            else:
                response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path})
            raise
        finally:
            request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = rid
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    app.include_router(health.router)
    app.include_router(stats.router)
    app.include_router(search.router)
    app.include_router(users.router)
    app.include_router(chat.router)
    return app


app = create_app()
