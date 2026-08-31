"""Tests for reporting backend exceptions to Sentry.

Visitors do not report failures — they close the tab. The container that failed
scaled back to zero and its logs aged out, so a production exception that is not
pushed somewhere durable is simply never seen. These tests pin the three
things that make it worth having: it is off unless a DSN is configured, it
fires before the visitor's error line is emitted, and it stays quiet for the
outcomes that are normal operation rather than bugs.

Nothing here may reach a real Sentry endpoint. Two things guarantee that: the
suite runs with no `SENTRY_DSN` (see `scripts/backend-test.sh`), and every test
that turns reporting on patches both the SDK's `init` and its capture function,
so no client and no transport is ever built.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import errors
from tests.test_chat import _failing_groq, _groq_quota_error, _mock_groq

# Well-formed but unroutable: `.invalid` is reserved (RFC 2606) and can never
# resolve, so even a bug that built a real transport could not phone home.
FAKE_DSN = "https://publickey@o0.ingest.sentry.invalid/1"

VISITOR = "203.0.113.77"
PER_MINUTE_LIMIT = 10


@pytest.fixture(autouse=True)
def reporting_disabled_by_default():
    """Every test starts (and leaves) reporting off, as the suite runs it."""
    errors.init_error_reporting(None)
    yield
    errors.init_error_reporting(None)


@pytest.fixture
def reporting_enabled():
    """Turn reporting on without building a client, and yield the capture mock."""
    with patch("app.errors.sentry_sdk.init") as init:
        errors.init_error_reporting(FAKE_DSN)
    assert init.called, "the fixture did not actually enable reporting"
    with patch("app.errors.sentry_sdk.capture_exception") as capture:
        yield capture


def _chat(client, **kwargs):
    return client.post("/api/chat", json={"message": "Hello"}, **kwargs)


# ---------------------------------------------------------------------------
# The DSN switch
# ---------------------------------------------------------------------------


def test_no_dsn_initialises_nothing():
    """Absent locally must mean absent, not a client pointed at nowhere."""
    with patch("app.errors.sentry_sdk.init") as init:
        enabled = errors.init_error_reporting(None)

    assert enabled is False
    init.assert_not_called()


def test_a_dsn_initialises_reporting_once():
    with patch("app.errors.sentry_sdk.init") as init:
        enabled = errors.init_error_reporting(FAKE_DSN)

    assert enabled is True
    init.assert_called_once()
    assert init.call_args.kwargs["dsn"] == FAKE_DSN


def test_startup_initialises_reporting(mocked_app):
    """Reporting is armed once by application startup, not per request."""
    with patch("app.main.init_error_reporting") as init, TestClient(mocked_app):
        pass

    init.assert_called_once()


def test_nothing_is_sent_when_no_dsn_is_configured(client):
    """The state the suite and every developer machine runs in."""
    with (
        patch("app.errors.sentry_sdk.capture_exception") as capture,
        patch("app.llm.AsyncGroq", return_value=_failing_groq(RuntimeError("API down"))),
    ):
        response = _chat(client)

    assert response.status_code == 200
    capture.assert_not_called()


# ---------------------------------------------------------------------------
# What gets reported
# ---------------------------------------------------------------------------


def test_generic_failure_is_reported_with_its_traceback(client, reporting_enabled):
    with patch("app.llm.AsyncGroq", return_value=_failing_groq(RuntimeError("API down"))):
        response = _chat(client)

    assert response.status_code == 200
    reporting_enabled.assert_called_once()
    reported = reporting_enabled.call_args.args[0]
    assert isinstance(reported, RuntimeError)
    # The whole point of reporting is the traceback; an exception stripped of
    # one would arrive in the inbox saying nothing about where it came from.
    assert reported.__traceback__ is not None
    frames = []
    tb = reported.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_filename)
        tb = tb.tb_next
    assert any(frame.endswith("routes/chat.py") for frame in frames), frames


def test_provider_quota_exhaustion_is_not_reported(client, reporting_enabled):
    """The service-wide Groq allowance running out is an operating condition."""
    with patch("app.llm.AsyncGroq", return_value=_failing_groq(_groq_quota_error())):
        response = _chat(client)

    assert json.loads(response.text.strip().splitlines()[-1][2:]) == {"category": "provider_quota"}
    reporting_enabled.assert_not_called()


def test_a_rate_limited_visitor_is_not_reported(client, reporting_enabled):
    """A visitor meeting their own cap is the limiter working, not a bug."""
    with patch("app.llm.AsyncGroq", return_value=_mock_groq(["ok"])):
        for _ in range(PER_MINUTE_LIMIT):
            assert _chat(client, headers={"X-Client-IP": VISITOR}).status_code == 200
        rejected = _chat(client, headers={"X-Client-IP": VISITOR})

    assert rejected.status_code == 429
    reporting_enabled.assert_not_called()


def test_log_records_never_become_reports():
    """Every arm logs the traceback, so log-derived events would report them all.

    The SDK turns ERROR-level log records into events by default, which would
    silently report the provider-quota arm as well — the exact thing the arms
    exist to keep apart. Reporting must come only from the explicit capture.
    """
    with patch("app.errors.sentry_sdk.init") as init:
        errors.init_error_reporting(FAKE_DSN)

    kwargs = init.call_args.kwargs
    logging_integrations = [
        i for i in kwargs.get("integrations", []) if type(i).__name__ == "LoggingIntegration"
    ]
    assert logging_integrations, "no LoggingIntegration was configured"
    assert logging_integrations[0]._handler is None, "log records still become events"
    # Nor may the framework integrations capture anything on their own.
    assert kwargs.get("auto_enabling_integrations") is False


# ---------------------------------------------------------------------------
# Ordering, and the visitor's view
# ---------------------------------------------------------------------------


def test_reporting_happens_before_the_error_line_is_emitted(client):
    """A stream that closes mid-flight must not take the report down with it.

    Recorded at the two call sites inside the generator rather than by reading
    the response, because the body can be buffered before the test reads it —
    which would make a report emitted *after* the line look correct.
    """
    from app.routes import chat as chat_module

    events: list[str] = []
    real_error_line = chat_module._error_line

    def _record_error_line(category: str) -> str:
        events.append(f"error_line:{category}")
        return real_error_line(category)

    def _record_report(error: BaseException) -> None:
        events.append(f"report:{type(error).__name__}")

    with (
        patch("app.routes.chat._error_line", _record_error_line),
        patch("app.routes.chat.report_exception", _record_report),
        patch("app.llm.AsyncGroq", return_value=_failing_groq(RuntimeError("API down"))),
    ):
        response = _chat(client)

    assert response.status_code == 200
    assert events == ["report:RuntimeError", "error_line:generic"]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("psycopg: password authentication failed"), id="generic"),
        pytest.param(_groq_quota_error(), id="provider_quota"),
    ],
)
def test_the_visitor_sees_the_same_bytes_whether_reporting_is_on_or_off(client, error):
    """Reporting is an operator concern; it may not change the visitor's stream."""
    with patch("app.llm.AsyncGroq", return_value=_failing_groq(error)):
        off = _chat(client).content

    with patch("app.errors.sentry_sdk.init"):
        errors.init_error_reporting(FAKE_DSN)
    with (
        patch("app.errors.sentry_sdk.capture_exception"),
        patch("app.llm.AsyncGroq", return_value=_failing_groq(error)),
    ):
        on = _chat(client).content

    assert on == off
    # And still no internal detail in either mode.
    assert str(error).encode() not in on
    assert b"Traceback" not in on
