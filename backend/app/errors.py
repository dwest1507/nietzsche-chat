"""Reporting backend exceptions to Sentry, so a production failure is noticed.

Nobody tells us when the app breaks. A visitor who hits a failure closes the
tab, and the traceback goes to the platform log of a container that — because
the backend scales to zero — may no longer exist by the time anyone thinks to
look. Reporting pushes the traceback out to somewhere durable while it still
exists, and mails it to the operator unasked.

Two rules shape what is in here:

*Off unless configured.* `SENTRY_DSN` is absent locally and in CI, and that is
deliberate: development failures are already in front of us in the terminal,
and the free tier's monthly event budget exists to catch the production ones
nobody is around to report. No DSN means no client, no transport, and nothing
sent — which is also what keeps the test suite off the network.

*Only what we report explicitly.* The SDK would otherwise turn every
ERROR-level log record into an event, and every arm of the chat endpoint's
error handling logs one — including the arms that are ordinary operating
conditions rather than bugs. Both the log handler and the framework
auto-instrumentation are therefore switched off, leaving `report_exception` as
the single source of events.
"""

import logging

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

from .config import SENTRY_DSN

logger = logging.getLogger("uvicorn.error")

_reporting_enabled = False


def init_error_reporting(dsn: str | None = SENTRY_DSN) -> bool:
    """Arm error reporting if a DSN is configured. Returns whether it is armed."""
    global _reporting_enabled

    if not dsn:
        _reporting_enabled = False
        logger.info("SENTRY_DSN is unset; backend error reporting is disabled")
        return False

    sentry_sdk.init(
        dsn=dsn,
        # `event_level=None` installs no log handler, so log records never
        # become events; see the module docstring.
        integrations=[LoggingIntegration(event_level=None)],
        auto_enabling_integrations=False,
        # Questions and visitor addresses are not ours to send abroad, and
        # performance traces would spend the event budget on healthy requests.
        send_default_pii=False,
        traces_sample_rate=0.0,
    )
    _reporting_enabled = True
    logger.info("Backend error reporting enabled")
    return True


def report_exception(error: BaseException) -> None:
    """Report one exception, with its traceback. A no-op when unconfigured."""
    if not _reporting_enabled:
        return
    sentry_sdk.capture_exception(error)
