import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { GET } from '@/app/api/ready/route'

const SECRET = 'test-shared-secret'
const BACKEND_URL = 'http://backend.test'

function backendReadiness(status: string, httpStatus = 200) {
  return new Response(JSON.stringify({ status }), {
    status: httpStatus,
    headers: { 'Content-Type': 'application/json' },
  })
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  vi.stubEnv('BACKEND_SHARED_SECRET', SECRET)
  vi.stubEnv('CHAT_API_URL', BACKEND_URL)
  fetchMock = vi.fn(async () => backendReadiness('ready'))
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

describe('GET /api/ready', () => {
  it('attaches the shared secret header to the backend readiness call', async () => {
    await GET()

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe(`${BACKEND_URL}/api/ready`)
    expect(new Headers(init.headers).get('X-Backend-Secret')).toBe(SECRET)
  })

  it.each(['loading', 'ready', 'failed'])('reports the backend state %s', async (state) => {
    fetchMock.mockResolvedValue(backendReadiness(state))
    const response = await GET()

    expect(response.status).toBe(200)
    expect(await response.json()).toEqual({ status: state })
  })

  it('never leaks the secret into the response reaching the browser', async () => {
    const response = await GET()

    for (const [, value] of response.headers) {
      expect(value).not.toContain(SECRET)
    }
    expect(await response.text()).not.toContain(SECRET)
  })

  it('reports loading when the sleeping backend refuses the connection', async () => {
    // A scaled-to-zero container is exactly the case this endpoint exists for:
    // it is waking, not broken, so the client should keep polling.
    fetchMock.mockRejectedValue(new Error('ECONNREFUSED'))
    const response = await GET()

    expect(await response.json()).toEqual({ status: 'loading' })
  })

  it('reports a terminal failure when the backend rejects the probe', async () => {
    fetchMock.mockResolvedValue(new Response('Unauthorized', { status: 401 }))
    const response = await GET()

    expect(await response.json()).toEqual({ status: 'failed' })
  })

  it.each([502, 503, 504])(
    'reports loading when the platform edge answers %i for a sleeping container',
    async (httpStatus) => {
      // Railway's edge answers for a container that is still starting, so a 5xx
      // here is the cold start this endpoint exists to report — not a backend
      // that is up and broken. Latching `failed` would stop the poll and refuse
      // every later question until the page is reloaded.
      fetchMock.mockResolvedValue(new Response('Bad Gateway', { status: httpStatus }))
      const response = await GET()

      expect(await response.json()).toEqual({ status: 'loading' })
    }
  )

  it('reports a terminal failure when the backend answers with an unknown state', async () => {
    fetchMock.mockResolvedValue(backendReadiness('warming-up-ish'))
    const response = await GET()

    expect(await response.json()).toEqual({ status: 'failed' })
  })

  it('is never cached, so a poll sees the state now rather than at first wake', async () => {
    const response = await GET()

    expect(response.headers.get('Cache-Control')).toBe('no-store')
  })
})
