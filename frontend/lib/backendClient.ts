/**
 * Server-side client for the FastAPI backend — the only place the backend URL
 * and the shared secret live.
 *
 * The browser never talks to the backend directly: its Railway URL is public,
 * so every request must carry `X-Backend-Secret` to be served at all. Route
 * handlers go through `backendFetch` so no call can forget it, and so the
 * secret is never read anywhere it could reach the client.
 * See `docs/adr/0002-shared-secret-gateway.md`.
 */

const SECRET_HEADER = 'X-Backend-Secret'

export function backendUrl(): string {
  return process.env.CHAT_API_URL ?? 'http://localhost:8000'
}

/**
 * Fetch `path` (e.g. `/api/chat`) from the backend with the shared secret
 * attached, returning the raw `Response` for the caller to map or pipe.
 */
export function backendFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  headers.set(SECRET_HEADER, process.env.BACKEND_SHARED_SECRET ?? '')

  return fetch(`${backendUrl()}${path}`, { ...init, headers })
}
