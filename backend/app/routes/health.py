"""Liveness and readiness — two endpoints, deliberately not one.

`/api/health` is what Railway probes: open, flat, and silent about the models.
A healthcheck that waited for readiness would block every deploy on a full
model load, the stall the background warm-up exists to avoid.

`/api/ready` reports the warm-up state and is behind the shared secret. Neither
handler touches `get_pipeline()` — it holds the singleton lock for the whole
load. See `docs/adr/0003-readiness-is-separate-from-liveness.md`.
"""

from fastapi import APIRouter, Depends

from ..readiness import get_state
from ..security import require_shared_secret

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# No rate limit here, deliberately: the frontend polls this endpoint while the
# container wakes, and metering it would throttle the wake it is watching.
@router.get("/ready", dependencies=[Depends(require_shared_secret)])
async def ready() -> dict:
    return {"status": get_state()}
