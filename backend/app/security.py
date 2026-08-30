"""Shared-secret gateway for the endpoints only our frontend may call.

The backend is deployed to a public Railway URL and the browser never talks to
it directly — every chat request originates inside the Vercel route handler.
That handler presents `X-Backend-Secret`; anything else is an outsider and is
turned away before it can burn the Groq quota. See
`docs/adr/0002-shared-secret-gateway.md`.
"""

import hmac

from fastapi import Header, HTTPException, status

from .config import BACKEND_SHARED_SECRET

SECRET_HEADER = "X-Backend-Secret"


async def require_shared_secret(x_backend_secret: str = Header(default="")) -> None:
    """Reject any request that does not present the shared secret."""
    # Constant-time: a plain == leaks the secret one character at a time to a
    # caller who can measure the response. Compare as bytes — header values
    # arrive latin-1 decoded, and compare_digest rejects non-ASCII str, which
    # would turn an outsider's bad header into a 500 instead of a refusal.
    presented = x_backend_secret.encode("latin-1", "replace")
    if not hmac.compare_digest(presented, BACKEND_SHARED_SECRET.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
