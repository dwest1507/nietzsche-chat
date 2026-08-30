"""Tests for GET /api/ready — readiness reported apart from liveness.

Readiness is a separate, authenticated endpoint, and neither it nor
`/api/health` may reach the pipeline accessor: see
`docs/adr/0003-readiness-is-separate-from-liveness.md`.
"""

from unittest.mock import patch

from app import readiness
from tests.conftest import TEST_SHARED_SECRET, join_warm_up


def test_ready_reports_loading_while_the_models_load(client):
    join_warm_up()
    readiness.mark_loading()

    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "loading"}


def test_ready_reports_ready_once_the_pipeline_is_loaded(client):
    join_warm_up()
    readiness.mark_ready()

    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_reports_a_failed_warm_up_as_terminal_failure(client):
    join_warm_up()
    readiness.mark_failed()

    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "failed"}


def test_ready_without_the_secret_is_rejected(unauthenticated_client):
    response = unauthenticated_client.get("/api/ready")

    assert response.status_code == 401
    assert "status" not in response.json()


def test_ready_with_a_wrong_secret_is_rejected(unauthenticated_client):
    response = unauthenticated_client.get("/api/ready", headers={"X-Backend-Secret": "guess"})

    assert response.status_code == 401


def test_ready_never_calls_the_pipeline_accessor(client):
    """The accessor blocks on the singleton lock — probing through it would hang."""
    with patch("app.rag.pipeline.get_pipeline") as accessor:
        response = client.get("/api/ready")

    assert response.status_code == 200
    accessor.assert_not_called()


def test_ready_is_not_rate_limited(client):
    """The frontend polls this while the container wakes; a limit would break that."""
    join_warm_up()
    readiness.mark_loading()

    statuses = [
        client.get("/api/ready", headers={"X-Client-IP": "203.0.113.10"}).status_code
        for _ in range(15)
    ]

    assert statuses == [200] * 15
    assert TEST_SHARED_SECRET  # the client presents it on every one of those calls
