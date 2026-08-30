"""Tests for application startup and the readiness state it records.

Regression cover for the failure where the app was wedged at startup by a
stalled HuggingFace Hub request and never bound its port, so every frontend
request came back "backend is down" — and for the readiness split that reports
the warm-up while it is still in flight
(`docs/adr/0003-readiness-is-separate-from-liveness.md`).
"""

import threading
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app import main, readiness
from tests.conftest import TEST_SHARED_SECRET, join_warm_up

SECRET_HEADERS = {"X-Backend-Secret": TEST_SHARED_SECRET}


def get_off_thread(client: TestClient, path: str, timeout: float = 5.0):
    """GET `path` from a worker thread, so a blocked endpoint fails the test.

    A request that sat on the pipeline lock would otherwise hang the suite
    rather than report the regression it exists to catch.
    """
    answered: dict[str, object] = {}

    def call() -> None:
        answered["response"] = client.get(path)

    caller = threading.Thread(target=call, name="probe", daemon=True)
    caller.start()
    caller.join(timeout)
    assert "response" in answered, f"{path} did not answer while the models were loading"
    return answered["response"]


def test_startup_does_not_block_on_model_loading():
    """A slow model load must not stop the API from serving."""
    loading = threading.Event()
    finished = threading.Event()
    release = threading.Event()

    def slow_load():
        loading.set()
        release.wait(timeout=30)
        finished.set()
        return MagicMock()

    try:
        with patch("app.main.get_pipeline", side_effect=slow_load):
            from app.main import app

            with TestClient(app) as c:
                assert loading.wait(timeout=5), "warm-up never started"
                # The API answers while the load is still in flight.
                assert c.get("/api/health").status_code == 200
                assert not finished.is_set(), "startup waited for the model load"
    finally:
        release.set()


def test_warm_up_failure_does_not_break_startup():
    """If the models fail to load the app still starts, and says so terminally."""
    with patch("app.main.get_pipeline", side_effect=RuntimeError("hub unreachable")):
        from app.main import app

        with TestClient(app, headers=SECRET_HEADERS) as c:
            assert c.get("/api/health").status_code == 200

            join_warm_up()
            # Terminal failure, never perpetual loading: a frontend told
            # "loading" by a permanently broken pipeline would poll forever.
            assert c.get("/api/ready").json() == {"status": "failed"}


def test_warm_up_records_loading_then_ready():
    """The warm-up thread, not the pipeline accessor, is what records the state."""
    observed: list[str] = []

    def load() -> MagicMock:
        observed.append(readiness.get_state())
        return MagicMock()

    # Start from a state the warm-up must overwrite, so a no-op would fail here.
    readiness.mark_failed()

    with patch("app.main.get_pipeline", side_effect=load):
        main._warm_pipeline()

    assert observed == ["loading"]
    assert readiness.get_state() == "ready"


def test_warm_up_failure_records_terminal_failure():
    """A raising warm-up records `failed`, not the loading state it started in."""
    with patch("app.main.get_pipeline", side_effect=RuntimeError("hub unreachable")):
        main._warm_pipeline()

    assert readiness.get_state() == "failed"


def test_health_and_readiness_answer_while_the_pipeline_lock_is_held():
    """Neither endpoint may block on the singleton lock the warm-up holds.

    `get_pipeline()` holds that lock for the whole model load, so an endpoint
    that probed through it would hang for exactly the duration it exists to
    report on. Both must answer mid-load, and readiness must say `loading`.
    """
    from app.rag import pipeline as pipeline_module

    loading = threading.Event()
    release = threading.Event()

    def slow_load() -> MagicMock:
        # Hold the real singleton lock, exactly as a real model load does.
        with pipeline_module._pipeline_lock:
            loading.set()
            release.wait(timeout=30)
        return MagicMock()

    try:
        with patch("app.main.get_pipeline", side_effect=slow_load):
            from app.main import app

            with TestClient(app, headers=SECRET_HEADERS) as c:
                assert loading.wait(timeout=5), "warm-up never started"

                health = get_off_thread(c, "/api/health")
                ready = get_off_thread(c, "/api/ready")

                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                assert ready.status_code == 200
                assert ready.json() == {"status": "loading"}

                release.set()
                join_warm_up()
                assert get_off_thread(c, "/api/ready").json() == {"status": "ready"}
    finally:
        release.set()
