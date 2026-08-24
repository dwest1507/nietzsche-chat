import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import useNietzscheChat from '@/lib/useNietzscheChat'

const SOURCES = [
  {
    title: 'Thus Spake Zarathustra',
    translator: 'Thomas Common',
    url: 'https://www.gutenberg.org/',
    text: 'I teach you the Superman.',
  },
]

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

describe('useNietzscheChat', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('streams a full round-trip: sources, tokens, done', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        streamResponse([
          `2:${JSON.stringify(SOURCES)}\n`,
          '0:"I teach"\n',
          '0:" you the Superman."\n',
          'd:{"finishReason": "stop"}\n',
        ])
      )
    vi.stubGlobal('fetch', fetchMock)

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
    const fetchMock = vi
      .fn()
      .mockResolvedValue(streamResponse(['0:"one"\n', 'd:{"finishReason": "stop"}\n']))
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('First question'))
    await waitFor(() => expect(result.current.status).toBe('idle'))

    fetchMock.mockResolvedValue(streamResponse(['0:"two"\n', 'd:{"finishReason": "stop"}\n']))
    act(() => result.current.sendMessage('Follow-up'))
    await waitFor(() => expect(result.current.status).toBe('idle'))

    const secondCall = JSON.parse(fetchMock.mock.calls[1][1].body as string)
    expect(secondCall.message).toBe('Follow-up')
    expect(secondCall.history).toEqual([
      { role: 'user', content: 'First question' },
      { role: 'assistant', content: 'one' },
    ])
  })

  it('sets an error state on a failed response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('nope', { status: 502 })))

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))

    await waitFor(() => expect(result.current.status).toBe('error'))
    expect(result.current.errorMessage).toMatch(/unreachable/)
    // The empty assistant placeholder is removed; the user message stays
    expect(result.current.messages).toHaveLength(1)
  })

  it('uses a rate-limit message on 429', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('slow down', { status: 429 })))

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))

    await waitFor(() => expect(result.current.status).toBe('error'))
    expect(result.current.errorMessage).toMatch(/wait a minute/)
  })

  it('sets an error state on a stream error event', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          streamResponse([`2:${JSON.stringify(SOURCES)}\n`, '3:"Generation failed"\n'])
        )
    )

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))

    await waitFor(() => expect(result.current.status).toBe('error'))
  })

  it('clear resets messages and status', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(streamResponse(['0:"hi"\n', 'd:{"finishReason": "stop"}\n']))
    )

    const { result } = renderHook(() => useNietzscheChat())
    act(() => result.current.sendMessage('Hello'))
    await waitFor(() => expect(result.current.status).toBe('idle'))
    expect(result.current.messages.length).toBeGreaterThan(0)

    act(() => result.current.clear())
    expect(result.current.messages).toHaveLength(0)
    expect(result.current.status).toBe('idle')
  })
})
