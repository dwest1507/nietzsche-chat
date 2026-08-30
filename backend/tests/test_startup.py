"""Tests for application startup.

Regression cover for the failure where the app was wedged at startup by a
stalled HuggingFace Hub request and never bound its port, so every frontend
request came back "backend is down".
"""

import threading
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


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
    """If the models fail to load, the app still starts (and logs the failure)."""
    with patch("app.main.get_pipeline", side_effect=RuntimeError("hub unreachable")):
        from app.main import app

        with TestClient(app) as c:
            assert c.get("/api/health").status_code == 200
