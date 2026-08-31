# The Next.js route is the only client of the backend

The browser never talks to the FastAPI backend; every chat request originates inside the
Vercel route handler that proxies to it. Two things followed from that in production which
did not show up in development, where the browser hits `:8000` directly:

- The backend's per-IP rate limit collapsed. `get_remote_address` sees a rotating Vercel
  egress IP, never a visitor, so one bucket was shared by everyone — unfair lockouts and
  no real ceiling at the same time.
- The Railway URL is public and unauthenticated, so anyone who found it could POST
  `/api/chat` directly and burn the Groq daily quota the limit exists to protect.

So the two services share a secret (`BACKEND_SHARED_SECRET`, set identically in Railway
and Vercel). The route handler forwards the visitor's address as `X-Client-IP` alongside
the secret header; the backend rejects any request without a valid secret and only then
keys its rate limiter on the forwarded address. Checking the secret first is what stops
`X-Client-IP` from being trivially spoofable by an outsider.

## Consequences

- The secret must be rotated in two places. If Railway and Vercel drift, every chat
  request fails — a failure mode the previous open backend could not have.
- `ALLOWED_ORIGINS` / CORS is now vestigial: no browser request ever reaches the backend,
  so it guards a path nothing uses. It stays set as defence in depth, but it is **not**
  what protects the backend, and nothing should be built on the assumption that it is.
