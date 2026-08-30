"""FastAPI application entry point."""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from . import readiness
from .config import ALLOWED_ORIGINS
from .rag.pipeline import get_pipeline
from .ratelimit import limiter
from .routes.chat import router as chat_router
from .routes.health import router as health_router

logger = logging.getLogger("uvicorn.error")


def _warm_pipeline() -> None:
    # This thread is the only writer of the readiness state; /api/ready reads
    # it rather than the pipeline, which would block on the load below.
    readiness.mark_loading()
    try:
        get_pipeline()
        readiness.mark_ready()
        logger.info("RAG pipeline ready")
    except Exception:
        # Terminal for readiness: a pipeline that cannot load must not read as
        # "still loading", or the frontend waits for a wake that never comes.
        # The first chat request still retries the load on its own.
        readiness.mark_failed()
        logger.exception("RAG pipeline warm-up failed; the first chat request will retry")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load models so the first request isn't slow, but off the startup path:
    # loading takes tens of seconds, and on a cold cache it hits the network.
    # Doing it inline here means a slow or stalled load stops the API from ever
    # binding its port, which reads to the frontend as "backend is down".
    threading.Thread(target=_warm_pipeline, name="pipeline-warmup", daemon=True).start()
    yield


app = FastAPI(title="Nietzsche Chat API", lifespan=lifespan)

# Rate limiting. The limit itself is applied by the decorator on the chat
# endpoint, deliberately and not by SlowAPIMiddleware: middleware runs before
# route dependencies, so it would key the limiter on a forwarded address that
# the shared-secret check has not yet vouched for. See app/ratelimit.py.
# app.state.limiter is what the RateLimitExceeded handler reads to shape the 429.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(health_router, prefix="/api")
