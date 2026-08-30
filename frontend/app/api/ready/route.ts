import { backendFetch } from '@/lib/backendClient'
import { isReadinessState } from '@/lib/readiness'
import type { ReadinessState } from '@/lib/types'

function readiness(status: ReadinessState): Response {
  // The client polls this while the container wakes, so a cached answer would
  // report the state at first wake forever.
  return Response.json({ status }, { headers: { 'Cache-Control': 'no-store' } })
}

/**
 * The browser's view of whether the backend can answer a question yet.
 *
 * The backend scales to zero and loads its models on a background thread, so
 * the chat pings this on mount — which is what wakes the container — and polls
 * while it reports `loading`. backendFetch owns the URL and the shared secret;
 * neither may reach the browser. See
 * `docs/adr/0003-readiness-is-separate-from-liveness.md`.
 */
export async function GET() {
  let backendResponse: Response
  try {
    backendResponse = await backendFetch('/api/ready')
  } catch {
    // A sleeping container refuses the connection: it is waking, not broken,
    // so report it as still loading and let the caller keep polling.
    return readiness('loading')
  }

  if (!backendResponse.ok) {
    // A backend that is up and refusing us (a wrong secret, a 500) will not
    // become ready by being asked again — report it terminally rather than
    // leaving the chat polling for a readiness that never arrives.
    return readiness('failed')
  }

  let status: unknown
  try {
    status = ((await backendResponse.json()) as { status?: unknown }).status
  } catch {
    status = null
  }

  return readiness(isReadinessState(status) ? status : 'failed')
}
