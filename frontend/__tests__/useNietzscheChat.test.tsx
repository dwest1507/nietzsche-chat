import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import useNietzscheChat from '@/lib/useNietzscheChat'
import type { ChatStatus } from '@/lib/types'

const SOURCES = [
  {
    title: 'Thus Spake Zarathustra',
    translator: 'Thomas Common',
    url: 'https://www.gutenberg.org/',
    text: 'I teach you the Superman.',
  },
]

// The hook polls readiness on this cadence; see lib/useNietzscheChat.ts.
const POLL_MS = 2000

function streamResponse(lines: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) controller.enqueue(encoder.encode(line))
      controller.close()
    },
  })
  return new Response(body, { status })
}

function readyResponse(status: string): Response {
  return new Response(JSON.stringify({ status }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubRoutes(handler: (...args: Parameters<typeof fetch>) => Promise<Response>) {
  const fetchMock = vi.fn(handler)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/**
 * A fetch stub answering both endpoints the hook talks to: the readiness ping
 * it makes on mount, and the chat call under test.
 */
function stubFetch(handlers: { chat?: () => Response; readiness?: () => Response } = {}) {
  const chat = handlers.chat ?? (() => streamResponse(['d:{"finishReason": "stop"}\n']))
  const readiness = handlers.readiness ?? (() => readyResponse('ready'))
  return stubRoutes(async (...args: Parameters<typeof fetch>) =>
    String(args[0]) === '/api/ready' ? readiness() : chat()
  )
}

type FetchMock = ReturnType<typeof stubFetch>

function callsTo(fetchMock: FetchMock, path: string) {
  return fetchMock.mock.calls.filter((call) => String(call[0]) === path)
}

describe('useNietzscheChat', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('streams a full round-trip: sources, tokens, done', async () => {
    stubFetch({
      chat: () =>
        streamResponse([
          `2:${JSON.stringify(SOURCES)}\n`,
          '0:"I teach"\n',
          '0:" you the Superman."\n',
          'd:{"finishReason": "stop"}\n',
        ]),
    })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('What is the Übermensch?'))

    await waitFor(() => expect(result.current.status).toBe('idle'))

    expect(result.current.messages).toHaveLength(2)
    const [user, assistant] = result.current.messages
    expect(user).toMatchObject({ role: 'user', content: 'What is the Übermensch?' })
    expect(assistant).toMatchObject({
      role: 'assistant',
      content: 'I teach you the Superman.',
      sources: SOURCES,
    })
  })

  it('sends prior messages as history', async () => {
    const answers = [
      () => streamResponse(['0:"one"\n', 'd:{"finishReason": "stop"}\n']),
      () => streamResponse(['0:"two"\n', 'd:{"finishReason": "stop"}\n']),
    ]
    let answered = 0
    const fetchMock = stubFetch({ chat: () => answers[answered++]() })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('First question'))
    await waitFor(() => expect(result.current.status).toBe('idle'))

    act(() => result.current.sendMessage('Follow-up'))
    await waitFor(() => expect(result.current.status).toBe('idle'))

    const secondCall = JSON.parse(callsTo(fetchMock, '/api/chat')[1][1]!.body as string)
    expect(secondCall.message).toBe('Follow-up')
    expect(secondCall.history).toEqual([
      { role: 'user', content: 'First question' },
      { role: 'assistant', content: 'one' },
    ])
  })

  it('sets an error state on a failed response', async () => {
    stubFetch({ chat: () => new Response('nope', { status: 502 }) })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))

    await waitFor(() => expect(result.current.status).toBe('error'))
    expect(result.current.errorMessage).toMatch(/unreachable/)
    // The empty assistant placeholder is removed; the user message stays
    expect(result.current.messages).toHaveLength(1)
  })

  it('uses a rate-limit message on 429', async () => {
    stubFetch({ chat: () => new Response('slow down', { status: 429 }) })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))

    await waitFor(() => expect(result.current.status).toBe('error'))
    expect(result.current.errorMessage).toMatch(/wait a minute/)
  })

  it('sets an error state on a stream error event', async () => {
    stubFetch({
      chat: () => streamResponse([`2:${JSON.stringify(SOURCES)}\n`, '3:"Generation failed"\n']),
    })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))

    await waitFor(() => expect(result.current.status).toBe('error'))
  })

  it('clear resets messages and status', async () => {
    stubFetch({ chat: () => streamResponse(['0:"hi"\n', 'd:{"finishReason": "stop"}\n']) })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))
    await waitFor(() => expect(result.current.status).toBe('idle'))
    expect(result.current.messages.length).toBeGreaterThan(0)

    act(() => result.current.clear())
    expect(result.current.messages).toHaveLength(0)
    expect(result.current.status).toBe('idle')
  })
})

describe('useNietzscheChat — readiness', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  /** Let the pending readiness fetch settle without waiting on the clock. */
  async function settle(ms = 0) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms)
    })
  }

  it('pings readiness when the chat mounts, waking the backend while the visitor reads', async () => {
    const fetchMock = stubFetch()

    renderHook(() => useNietzscheChat())
    await settle()

    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(1)
  })

  it('polls readiness while the backend reports it is still loading', async () => {
    const fetchMock = stubFetch({ readiness: () => readyResponse('loading') })

    renderHook(() => useNietzscheChat())
    await settle()
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(1)

    await settle(POLL_MS)
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(2)

    await settle(POLL_MS)
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(3)
  })

  it('reports the waking status while the backend is not ready', async () => {
    stubFetch({ readiness: () => readyResponse('loading') })

    const { result } = renderHook(() => useNietzscheChat())
    await settle()

    expect(result.current.status).toBe('waking')
  })

  it('stops polling, and stops waking, once the backend reports ready', async () => {
    const states = ['loading', 'ready']
    let asked = 0
    const fetchMock = stubFetch({ readiness: () => readyResponse(states[asked++] ?? 'ready') })

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    expect(result.current.status).toBe('waking')

    await settle(POLL_MS)
    expect(result.current.status).toBe('idle')
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(2)

    await settle(POLL_MS * 3)
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(2)
  })

  it('stops polling when the backend reports a terminal failure', async () => {
    const fetchMock = stubFetch({ readiness: () => readyResponse('failed') })

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(1)

    await settle(POLL_MS * 5)
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(1)
    // A pipeline that failed to load will never wake; don't claim it is stirring.
    expect(result.current.status).not.toBe('waking')
  })

  it('never shows the waking status to a visitor arriving at a warm backend', async () => {
    stubFetch({ readiness: () => readyResponse('ready') })

    const seen: ChatStatus[] = []
    renderHook(() => {
      const chat = useNietzscheChat()
      seen.push(chat.status)
      return chat
    })
    await settle(POLL_MS * 2)

    expect(seen).not.toContain('waking')
    expect(seen.at(-1)).toBe('idle')
  })

  it('keeps polling when the readiness route itself cannot be reached', async () => {
    const fetchMock = stubRoutes(async (...args: Parameters<typeof fetch>) => {
      if (String(args[0]) === '/api/ready') throw new Error('offline')
      return streamResponse([])
    })

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    expect(result.current.status).toBe('waking')

    await settle(POLL_MS)
    expect(callsTo(fetchMock, '/api/ready')).toHaveLength(2)
  })
})
