"""Tests for GET /api/health."""


def test_health_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_says_nothing_about_the_models(unauthenticated_client):
    """Liveness stays flat and silent whatever the pipeline is doing.

    Railway probes this endpoint. If it reported model state, every deploy
    would wait on a full model load — the stall the warm-up thread exists to
    avoid. See docs/adr/0003-readiness-is-separate-from-liveness.md.
    """
    from app import readiness

    for state in (readiness.mark_loading, readiness.mark_failed, readiness.mark_ready):
        state()
        response = unauthenticated_client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
