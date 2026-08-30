"""FastAPI application entry point."""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .config import ALLOWED_ORIGINS
from .rag.pipeline import get_pipeline
from .routes.chat import router as chat_router
from .routes.health import router as health_router

logger = logging.getLogger("uvicorn.error")


def _warm_pipeline() -> None:
    try:
        get_pipeline()
        logger.info("RAG pipeline ready")
    except Exception:
        logger.exception("RAG pipeline warm-up failed; the first chat request will retry")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load models so the first request isn't slow, but off the startup path:
    # loading takes tens of seconds, and on a cold cache it hits the network.
    # Doing it inline here means a slow or stalled load stops the API from ever
    # binding its port, which reads to the frontend as "backend is down".
    threading.Thread(target=_warm_pipeline, name="pipeline-warmup", daemon=True).start()
    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Nietzsche Chat API", lifespan=lifespan)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(health_router, prefix="/api")
