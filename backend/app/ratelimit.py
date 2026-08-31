"""The one rate limiter, keyed on the visitor address the frontend forwards.

The browser never talks to this service: every chat request arrives from the
Vercel route handler, so the connecting address is a rotating egress address
shared by everyone. Keying on it gave one bucket to the whole internet — an
unfair lockout and no real per-visitor ceiling at the same time. The route
handler therefore forwards the visitor's address as `X-Client-IP` and we key on
that instead, falling back to the connecting address for local development,
where the browser does hit `:8000` directly and nothing adds the header.

That header is trusted *only* because `require_shared_secret` has already run
when we read it. The limit is enforced by the `@limiter.limit` decorator on the
endpoint, which wraps the endpoint function and so runs after its dependencies;
`SlowAPIMiddleware` would run before them and let an outsider's spoofed
`X-Client-IP` drain a stranger's bucket on the way to being rejected. See
`docs/adr/0002-shared-secret-gateway.md`.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

CLIENT_ADDRESS_HEADER = "X-Client-IP"

# The day limit is the real guard on the Groq quota — 100 answers is far more
# than any reader needs. The minute limit only smooths bursts, so one visitor
# cannot spend a whole day's allowance in a few seconds.
CHAT_RATE_LIMIT = "10/minute;100/day"


def visitor_address(request: Request) -> str:
    """The address to bill this request to: the forwarded one when we have it."""
    return request.headers.get(CLIENT_ADDRESS_HEADER) or get_remote_address(request)


limiter = Limiter(key_func=visitor_address)
