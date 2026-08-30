"""Tests for the per-visitor rate limit on /api/chat.

The browser never reaches the backend directly, so the connecting address is a
rotating Vercel egress address rather than a visitor. The route handler forwards
the visitor's address as `X-Client-IP` and the limiter keys on that — but only
once the shared secret has been checked, or an outsider could spoof the header
and consume (or poison) any visitor's bucket. See
`docs/adr/0002-shared-secret-gateway.md`.
"""

from unittest.mock import patch

import pytest

from tests.test_chat import _mock_groq

PER_MINUTE_LIMIT = 10

VISITOR = "203.0.113.10"
OTHER_VISITOR = "198.51.100.20"


@pytest.fixture(autouse=True)
def mocked_groq():
    """Every chat request in this module is served from a stubbed Groq client."""
    with patch("app.llm.AsyncGroq", return_value=_mock_groq(["The", " will", " to", " power"])):
        yield


def _chat(client, address: str | None = None):
    headers = {"X-Client-IP": address} if address else {}
    return client.post(
        "/api/chat",
        json={"message": "What is the will to power?"},
        headers=headers,
    )


def test_the_same_forwarded_address_consumes_one_bucket_across_requests(client):
    for _ in range(PER_MINUTE_LIMIT):
        assert _chat(client, VISITOR).status_code == 200

    assert _chat(client, VISITOR).status_code == 429


def test_two_forwarded_addresses_consume_separate_buckets(client):
    for _ in range(PER_MINUTE_LIMIT):
        assert _chat(client, VISITOR).status_code == 200
    assert _chat(client, VISITOR).status_code == 429

    # A different visitor is untouched by the first one's exhausted bucket.
    for _ in range(PER_MINUTE_LIMIT):
        assert _chat(client, OTHER_VISITOR).status_code == 200


def test_exceeding_the_per_minute_limit_returns_a_rate_limited_response(client):
    for _ in range(PER_MINUTE_LIMIT):
        _chat(client, VISITOR)

    response = _chat(client, VISITOR)

    assert response.status_code == 429
    assert "10 per 1 minute" in response.json()["error"]


def test_the_chat_route_limits_visitors_per_minute_and_per_day():
    """The day limit is the real quota guard; the minute limit only smooths bursts."""
    from app.ratelimit import limiter
    from app.routes.chat import chat

    name = f"{chat.__module__}.{chat.__name__}"
    limits = {str(limit.limit) for limit in limiter._route_limits[name]}

    assert limits == {"10 per 1 minute", "100 per 1 day"}


def test_a_request_without_a_forwarded_address_falls_back_to_the_connecting_address(client):
    """Local development hits the backend directly, with no proxy to add the header."""
    for _ in range(PER_MINUTE_LIMIT):
        assert _chat(client).status_code == 200

    assert _chat(client).status_code == 429


def test_a_forwarded_address_without_a_valid_secret_is_rejected_and_never_keyed(
    unauthenticated_client, client
):
    """The trap: the limiter must run after the secret check, not before it.

    If a spoofed `X-Client-IP` were keyed before the request was rejected, an
    outsider could drain any visitor's allowance without ever being served.
    """
    assert _chat(unauthenticated_client, VISITOR).status_code == 401

    # The rejected request consumed nothing: the visitor still has all of it.
    for _ in range(PER_MINUTE_LIMIT):
        assert _chat(client, VISITOR).status_code == 200


def test_the_per_visitor_cap_is_rejected_before_the_stream_starts(client):
    """The cap is enforced by the decorator, so it never reaches the generator.

    That is why it travels as an HTTP 429 rather than as a `3:` category: there
    is no stream to put a category on yet.
    """
    for _ in range(PER_MINUTE_LIMIT):
        _chat(client, VISITOR)

    response = _chat(client, VISITOR)

    assert response.status_code == 429
    assert "3:" not in response.text
    assert "2:" not in response.text


def test_the_per_visitor_cap_is_distinguishable_from_provider_quota_exhaustion(client):
    """One visitor's exhausted allowance must not read as the service's."""
    from tests.test_chat import _failing_groq, _groq_quota_error

    with patch("app.llm.AsyncGroq", return_value=_failing_groq(_groq_quota_error())):
        provider_exhausted = _chat(client, OTHER_VISITOR)

    assert provider_exhausted.status_code == 200
    assert '"category": "provider_quota"' in provider_exhausted.text

    for _ in range(PER_MINUTE_LIMIT):
        _chat(client, VISITOR)
    capped = _chat(client, VISITOR)

    assert capped.status_code == 429
    assert "provider_quota" not in capped.text
