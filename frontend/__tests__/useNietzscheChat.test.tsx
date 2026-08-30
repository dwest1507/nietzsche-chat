import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { StrictMode } from 'react'
import { renderHook, act, waitFor } from '@testing-library/react'
import useNietzscheChat from '@/lib/useNietzscheChat'
import type { ChatStatus, ReadinessState } from '@/lib/types'

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

/**
 * Let pending work settle without waiting on the wall clock: flushes the
 * microtasks a fetch resolves through, and any timer due within `ms`.
 */
async function settle(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('useNietzscheChat — readiness', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

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

describe('useNietzscheChat — holding a question sent while the backend wakes', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  /**
   * A readiness stub the test drives by hand: it answers whatever state the
   * returned handle currently holds, so a test can wake (or fail) the backend
   * mid-flight rather than counting poll responses.
   */
  function readinessDial(initial: ReadinessState = 'loading') {
    const dial = { state: initial }
    const fetchMock = stubFetch({ readiness: () => readyResponse(dial.state) })
    return { dial, fetchMock }
  }

  it('fires no chat request when a question is sent while the backend is waking', async () => {
    const { fetchMock } = readinessDial('loading')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    expect(result.current.status).toBe('waking')

    act(() => result.current.sendMessage('What is the Übermensch?'))
    await settle()
    await settle(POLL_MS * 3)

    // Not a deferred request, not an aborted one: none at all.
    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(0)
  })

  it('holds a question sent before the first readiness answer arrives', async () => {
    // The headline case: a starter question clicked the instant the page loads
    // lands before /api/ready has answered at all. Readiness is still unknown,
    // so the backend may well be cold — the question must not be fired into it.
    let answerReadiness: (() => void) | null = null
    const pending = new Promise<void>((resolve) => {
      answerReadiness = resolve
    })
    const fetchMock = stubRoutes(async (...args: Parameters<typeof fetch>) => {
      if (String(args[0]) === '/api/ready') {
        await pending
        return readyResponse('ready')
      }
      return streamResponse(['d:{"finishReason": "stop"}\n'])
    })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('What is the Übermensch?'))
    await settle()

    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(0)

    await act(async () => {
      answerReadiness?.()
      await pending
    })
    await settle()

    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(1)
  })

  it('shows the held question in the transcript as accepted', async () => {
    readinessDial('loading')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    act(() => result.current.sendMessage('What is the Übermensch?'))
    await settle()

    expect(result.current.messages).toContainEqual(
      expect.objectContaining({ role: 'user', content: 'What is the Übermensch?' })
    )
  })

  it('dispatches the held question exactly once when readiness flips to ready', async () => {
    const { dial, fetchMock } = readinessDial('loading')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    act(() => result.current.sendMessage('What is the Übermensch?'))
    await settle()
    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(0)

    dial.state = 'ready'
    await settle(POLL_MS)

    const chatCalls = callsTo(fetchMock, '/api/chat')
    expect(chatCalls).toHaveLength(1)
    expect(JSON.parse(chatCalls[0][1]!.body as string).message).toBe('What is the Übermensch?')
  })

  it('does not dispatch the held question more than once when readiness is observed repeatedly', async () => {
    const { dial, fetchMock } = readinessDial('loading')

    // StrictMode re-runs effects, and the extra renders below make every
    // render-scoped dependency change identity — a dispatch keyed on the
    // readiness value alone fires again on each of those observations.
    const { result, rerender } = renderHook(() => useNietzscheChat(), { wrapper: StrictMode })
    await settle()
    act(() => result.current.sendMessage('Ask me once'))
    await settle()

    dial.state = 'ready'
    await settle(POLL_MS)
    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(1)

    rerender()
    await settle(POLL_MS * 3)
    rerender()
    await settle(POLL_MS * 3)

    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(1)
  })

  it('surfaces an error for a held question when warm-up fails terminally', async () => {
    const { dial, fetchMock } = readinessDial('loading')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    act(() => result.current.sendMessage('Will you wake?'))
    await settle()

    dial.state = 'failed'
    await settle(POLL_MS)

    // A pipeline that will never load must not leave the question hanging.
    expect(result.current.status).toBe('error')
    expect(result.current.errorMessage).toBeTruthy()
    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(0)
    expect(result.current.messages).toContainEqual(
      expect.objectContaining({ role: 'user', content: 'Will you wake?' })
    )
  })

  it('answers a question held through the cold start, with its source passages', async () => {
    const dial = { state: 'loading' as ReadinessState }
    stubFetch({
      readiness: () => readyResponse(dial.state),
      chat: () =>
        streamResponse([
          `2:${JSON.stringify(SOURCES)}\n`,
          '0:"I teach you the Superman."\n',
          'd:{"finishReason": "stop"}\n',
        ]),
    })

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    act(() => result.current.sendMessage('What is the Übermensch?'))
    await settle()

    dial.state = 'ready'
    await settle(POLL_MS)
    for (let i = 0; i < 20 && result.current.status !== 'idle'; i += 1) await settle()

    expect(result.current.status).toBe('idle')
    expect(result.current.messages).toHaveLength(2)
    expect(result.current.messages[1]).toMatchObject({
      role: 'assistant',
      content: 'I teach you the Superman.',
      sources: SOURCES,
    })
  })

  it('ignores a second question while one is held', async () => {
    const { dial, fetchMock } = readinessDial('loading')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    act(() => result.current.sendMessage('First'))
    await settle()
    act(() => result.current.sendMessage('Second'))
    await settle()

    expect(result.current.messages.filter((m) => m.role === 'user')).toHaveLength(1)

    dial.state = 'ready'
    await settle(POLL_MS)

    const chatCalls = callsTo(fetchMock, '/api/chat')
    expect(chatCalls).toHaveLength(1)
    expect(JSON.parse(chatCalls[0][1]!.body as string).message).toBe('First')
  })

  it('cancels a held question when the visitor presses stop', async () => {
    // ChatShell shows Stop while a question is held, so it has to do something:
    // a control that silently does nothing is worse than no control.
    const { dial, fetchMock } = readinessDial('loading')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    act(() => result.current.sendMessage('What is the Übermensch?'))
    await settle()
    expect(result.current.status).toBe('held')

    act(() => result.current.stop())
    await settle()
    expect(result.current.status).not.toBe('held')

    dial.state = 'ready'
    await settle(POLL_MS)
    await settle()

    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(0)
  })

  it('drops a held question when the chat is cleared', async () => {
    const { dial, fetchMock } = readinessDial('loading')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    act(() => result.current.sendMessage('Never mind'))
    await settle()
    act(() => result.current.clear())

    dial.state = 'ready'
    await settle(POLL_MS)

    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(0)
    expect(result.current.messages).toHaveLength(0)
  })

  it('sends against an already-warm backend immediately, without holding it', async () => {
    const { fetchMock } = readinessDial('ready')

    const { result } = renderHook(() => useNietzscheChat())
    await settle()
    expect(result.current.status).toBe('idle')

    act(() => result.current.sendMessage('Warm question'))

    // In flight on the same turn — no queue, no timer, no extra await.
    expect(callsTo(fetchMock, '/api/chat')).toHaveLength(1)
    expect(result.current.status).toBe('retrieving')
  })
})

describe('useNietzscheChat — telling the three failures apart', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const PROVIDER_QUOTA_STREAM = ['3:{"category":"provider_quota"}\n']
  const GENERIC_STREAM = ['3:{"category":"generic"}\n']

  /** Answer the first chat call with `first`, every later one with a good stream. */
  function stubFailThenAnswer(first: () => Response) {
    let answered = 0
    return stubFetch({
      chat: () =>
        answered++ === 0
          ? first()
          : streamResponse(['0:"Answered"\n', 'd:{"finishReason": "stop"}\n']),
    })
  }

  async function messageFor(chat: () => Response): Promise<string> {
    stubFetch({ chat })
    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))
    await waitFor(() => expect(result.current.status).toBe('error'))
    return result.current.errorMessage ?? ''
  }

  it('says the service is out of answers when the provider quota is spent', async () => {
    stubFetch({ chat: () => streamResponse(PROVIDER_QUOTA_STREAM) })

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))

    await waitFor(() => expect(result.current.status).toBe('error'))
    expect(result.current.errorMessage).toMatch(/today/i)
    // Not the visitor's own cap, and not the shrug of a generic failure.
    expect(result.current.errorMessage).not.toMatch(/wait a minute/)
    expect(result.current.errorMessage).not.toMatch(/unreachable/)
  })

  it('gives a distinct message for each of the three failures', async () => {
    const seen: string[] = []
    seen.push(await messageFor(() => streamResponse(PROVIDER_QUOTA_STREAM)))
    seen.push(await messageFor(() => new Response('slow down', { status: 429 })))
    seen.push(await messageFor(() => streamResponse(GENERIC_STREAM)))

    expect(seen.every((m) => m.length > 0)).toBe(true)
    expect(new Set(seen).size).toBe(3)
  })

  it.each([
    ['the provider quota is spent', () => streamResponse(PROVIDER_QUOTA_STREAM)],
    ['the visitor hits their own cap', () => new Response('slow down', { status: 429 })],
    ['something else breaks', () => streamResponse(GENERIC_STREAM)],
  ])('lets the visitor ask again after %s, without a reload', async (_label, first) => {
    stubFailThenAnswer(first)

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('First try'))
    await waitFor(() => expect(result.current.status).toBe('error'))

    act(() => result.current.sendMessage('Second try'))
    await waitFor(() => expect(result.current.status).toBe('idle'))

    expect(result.current.errorMessage).toBeNull()
    expect(result.current.messages.at(-1)).toMatchObject({
      role: 'assistant',
      content: 'Answered',
    })
  })
})
