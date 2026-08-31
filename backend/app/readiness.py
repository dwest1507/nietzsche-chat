"""Whether the retrieval pipeline can answer yet — tracked apart from it.

The app binds its port immediately and loads the models on a background
thread, so "the process is up" and "the process can answer" are different
states for tens of seconds after every wake. The warm-up thread records which
one we are in here, and `/api/ready` reads it.

Deliberately nothing in this module imports the pipeline. `get_pipeline()`
holds the singleton lock for the whole load, so a readiness probe routed
through it would block for exactly the duration it exists to report on. See
`docs/adr/0003-readiness-is-separate-from-liveness.md`.
"""

from typing import Literal

State = Literal["loading", "ready", "failed"]

# One writer (the warm-up thread) and many readers (request handlers), sharing
# a single immutable value. Rebinding a module global is atomic, so this needs
# no lock — and a lock here is the one thing that must not exist: it could park
# a request behind the load, which is the whole failure this module avoids.
_state: State = "loading"


def get_state() -> State:
    """What the warm-up thread last recorded."""
    return _state


def mark_loading() -> None:
    """The models are being loaded; requests that need them will be slow."""
    global _state
    _state = "loading"


def mark_ready() -> None:
    """The pipeline is loaded and can serve a question."""
    global _state
    _state = "ready"


def mark_failed() -> None:
    """The warm-up raised. Terminal: never report a broken pipeline as loading."""
    global _state
    _state = "failed"
