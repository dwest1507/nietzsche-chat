"""Tests for the shared-secret gateway that fronts /api/chat.

The Railway URL is public, so the only thing separating our Vercel route
handler from the rest of the internet is the `X-Backend-Secret` header.
"""

from unittest.mock import patch

from tests.conftest import TEST_SHARED_SECRET
from tests.test_chat import _mock_groq, _parse_lines


def test_chat_without_secret_header_is_rejected(unauthenticated_client):
    response = unauthenticated_client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 401


def test_chat_with_wrong_secret_is_rejected(unauthenticated_client):
    response = unauthenticated_client.post(
        "/api/chat",
        json={"message": "Hello"},
        headers={"X-Backend-Secret": TEST_SHARED_SECRET + "-wrong"},
    )
    assert response.status_code == 401


def test_rejected_chat_request_never_reaches_the_pipeline(unauthenticated_client, mock_pipeline):
    """Rejection must happen before any retrieval or generation work is done."""
    unauthenticated_client.post("/api/chat", json={"message": "Hello"})
    mock_pipeline.retrieve.assert_not_called()


def test_chat_with_correct_secret_is_served(client):
    tokens = ["The", " will", " to", " power", "."]
    with patch("app.llm.AsyncGroq", return_value=_mock_groq(tokens)):
        response = client.post("/api/chat", json={"message": "What is the will to power?"})

    assert response.status_code == 200
    assert _parse_lines(response.text, "0:") == tokens


def test_health_does_not_require_the_secret(unauthenticated_client):
    """Railway probes /api/health unauthenticated; it must stay open."""
    response = unauthenticated_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_non_ascii_secret_header_is_rejected_not_crashed(unauthenticated_client):
    """A raw byte above 0x7F must be a 401, not a 500.

    Header values arrive latin-1 decoded, so an outsider can hand us a
    non-ASCII str — and `hmac.compare_digest` refuses those, which would turn
    a rejection into an unhandled error (and, once Sentry is wired up, a
    stream of reports for what is really just a bad request).
    """
    response = unauthenticated_client.post(
        "/api/chat",
        json={"message": "Hello"},
        headers={"X-Backend-Secret": "café".encode("latin-1")},
    )
    assert response.status_code == 401
