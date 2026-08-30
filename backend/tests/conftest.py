"""Shared fixtures for the test suite."""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Set required env vars before importing the app
os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
os.environ.setdefault("BACKEND_SHARED_SECRET", "test_shared_secret")

TEST_SHARED_SECRET = os.environ["BACKEND_SHARED_SECRET"]
"""The secret the default `client` fixture presents on every request."""


SAMPLE_CHUNKS = [
    {
        "text": "The world is the will to power—and nothing besides! And ye yourselves are "
        "also this will to power—and nothing besides!",
        "work_id": "the_will_to_power_book_iii_and_iv",
        "title": "The Will to Power, Books III and IV",
        "translator": "Anthony M. Ludovici",
        "url": "https://www.gutenberg.org/",
    },
    {
        "text": "I teach you the Superman. Man is something that is to be surpassed. "
        "What have ye done to surpass man?",
        "work_id": "thus_spake_zarathustra",
        "title": "Thus Spake Zarathustra",
        "translator": "Thomas Common",
        "url": "https://www.gutenberg.org/",
    },
    {
        "text": "God is dead: but considering the state the species Man is in, there will "
        "perhaps be caves, for ages yet, in which his shadow will be shown.",
        "work_id": "the_joyful_wisdom",
        "title": "The Joyful Wisdom",
        "translator": "Thomas Common",
        "url": "https://www.gutenberg.org/",
    },
    {
        "text": "He who has a why to live for can bear almost any how. Suffering makes noble; "
        "it separates.",
        "work_id": "the_twilight_of_the_idols",
        "title": "The Twilight of the Idols",
        "translator": "Anthony M. Ludovici",
        "url": "https://www.gutenberg.org/",
    },
    {
        "text": "Master-morality and slave-morality. The noble type of man regards himself as "
        "a determiner of values. What is injurious to me is injurious in itself, he thinks.",
        "work_id": "beyond_good_and_evil",
        "title": "Beyond Good and Evil",
        "translator": "Helen Zimmern",
        "url": "https://www.gutenberg.org/",
    },
]


@pytest.fixture
def mock_pipeline():
    """RAGPipeline that returns sample chunk dicts without loading real models."""
    pipeline = MagicMock()
    pipeline.retrieve.return_value = SAMPLE_CHUNKS[:3]
    return pipeline


@pytest.fixture
def mocked_app(mock_pipeline):
    """The FastAPI app with the RAG pipeline (and so Groq's inputs) mocked out."""
    # Patch the pipeline in its home module and where the chat route imports it
    with (
        patch("app.rag.pipeline.get_pipeline", return_value=mock_pipeline),
        patch("app.routes.chat.get_pipeline", return_value=mock_pipeline),
        # `app.main` bound its own reference at import time, so patching the
        # home module alone would leave the startup warm-up thread reaching for
        # the real models — a download the suite must never make.
        patch("app.main.get_pipeline", return_value=mock_pipeline),
    ):
        from app.main import app

        yield app


@pytest.fixture
def unauthenticated_client(mocked_app):
    """TestClient that sends no shared-secret header — for the rejection cases."""
    with TestClient(mocked_app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def client(mocked_app):
    """TestClient with mocked RAG pipeline and Groq, presenting the shared secret."""
    with TestClient(
        mocked_app,
        raise_server_exceptions=True,
        headers={"X-Backend-Secret": TEST_SHARED_SECRET},
    ) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear the limiter's buckets so allowance never leaks between tests.

    The limiter is a module-level singleton, imported once for the whole
    session, so without this every request in the suite would draw down the
    same per-minute bucket and later tests would start seeing 429s.
    """
    from app.ratelimit import limiter

    limiter.reset()
    yield
    limiter.reset()


def join_warm_up(timeout: float = 10.0) -> None:
    """Wait out the lifespan warm-up thread rather than racing it.

    Starting the app starts a background thread that writes the readiness
    state; a test that asserts on (or pins) that state has to let it finish
    first. Joining the thread keeps that deterministic — no wall-clock sleeps.
    """
    for thread in threading.enumerate():
        if thread.name == "pipeline-warmup":
            thread.join(timeout)


@pytest.fixture(autouse=True)
def no_warm_up_backoff():
    """Take the wall-clock sleeps out of the warm-up's retry backoff.

    `_warm_pipeline` waits between attempts so a struggling hub is not hammered.
    That wait is real time the suite must never spend: the tests that exercise
    every attempt would each add the whole backoff. The delays themselves stay
    as they are, so a test can still assert what would have been waited.

    Replaces `app.main`'s own reference to the module, not `time.sleep` itself —
    patching the function would reach every test in the suite, including
    `test_retrieval_does_not_block_the_event_loop`, whose slow retrieval is a
    real sleep and which would then pass without proving anything.
    """
    with patch("app.main.time"):
        yield


@pytest.fixture(autouse=True)
def reset_readiness_state():
    """Keep the module-global readiness state from leaking between tests."""
    from app import readiness

    readiness.mark_loading()
    yield
    readiness.mark_loading()
