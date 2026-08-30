'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { isReadinessState } from './readiness'
import { createStreamParser } from './streamParser'
import type { ChatMessageData, ChatStatus, ReadinessState, Source } from './types'

const HISTORY_LIMIT = 10

// Fast enough that a woken backend is noticed almost as soon as it is ready,
// slow enough not to hammer the route while tens of seconds of models load.
const READINESS_POLL_MS = 2000

const GENERIC_ERROR =
  'Nietzsche is unreachable at the moment. Please check your connection and try again.'
const RATE_LIMIT_ERROR =
  'Even Zarathustra needed rest — too many questions at once. Please wait a minute and try again.'
// Draft copy, in the app's voice: the warm-up failed terminally, so a held
// question will never be answered and must not wait for a wake that never comes.
const WARMUP_FAILED_ERROR =
  'Nietzsche could not be roused — his library failed to load. Please try again in a few minutes.'

let nextId = 0
function makeId(): string {
  nextId += 1
  return `msg-${nextId}`
}

interface ChatTurn {
  role: ChatMessageData['role']
  content: string
}

/** A question accepted from the visitor but not yet sent to a waking backend. */
interface HeldQuestion {
  text: string
  history: ChatTurn[]
}

export interface NietzscheChat {
  messages: ChatMessageData[]
  status: ChatStatus
  errorMessage: string | null
  sendMessage: (text: string) => void
  stop: () => void
  clear: () => void
}

export default function useNietzscheChat(): NietzscheChat {
  const [messages, setMessages] = useState<ChatMessageData[]>([])
  const [chatStatus, setChatStatus] = useState<ChatStatus>('idle')
  // `null` until the first readiness answer arrives — an already-warm backend
  // goes straight to `ready`, so its visitor never sees the waking state.
  const [readiness, setReadiness] = useState<ReadinessState | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  // A question sent while the backend is still waking is held here instead of
  // being fired into a cold pipeline, where it would block on the singleton
  // model lock for the rest of the load and read as an unexplained hang.
  // See docs/adr/0001-scale-to-zero-with-warm-on-arrival.md.
  const heldRef = useRef<HeldQuestion | null>(null)
  // Bumped whenever a question is held, so the dispatch effect below also runs
  // for a backend that turned out to be ready already.
  const [heldCount, setHeldCount] = useState(0)

  // The backend scales to zero and loads its models on a background thread, so
  // ping readiness as the chat mounts: the ping is what wakes the container,
  // and the models load while the visitor is still reading. Keep asking while
  // it reports loading. See docs/adr/0001 and docs/adr/0003.
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null

    async function poll(): Promise<void> {
      // A route or network hiccup reads as "not yet" — the next poll retries.
      let state: ReadinessState = 'loading'
      try {
        const response = await fetch('/api/ready')
        const body = (await response.json()) as { status?: unknown }
        if (response.ok && isReadinessState(body.status)) state = body.status
      } catch {
        // Left as 'loading': a sleeping container is waking, not broken.
      }
      if (cancelled) return

      setReadiness(state)
      // Stop once ready, and stop on a terminal failure: a pipeline that could
      // not load will never become ready, so polling it would never end.
      if (state === 'loading') {
        timer = setTimeout(() => void poll(), READINESS_POLL_MS)
      }
    }

    void poll()

    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [])

  // Waking is an idle-chat state: it says the backend cannot answer yet, and
  // must never mask the progress of a message already in flight.
  const status: ChatStatus =
    chatStatus === 'idle' && readiness === 'loading' ? 'waking' : chatStatus

  const updateMessage = useCallback(
    (id: string, patch: (m: ChatMessageData) => ChatMessageData) => {
      setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)))
    },
    []
  )

  const removeIfEmpty = useCallback((id: string) => {
    setMessages((prev) => prev.filter((m) => m.id !== id || m.content.length > 0))
  }, [])

  // Everything from the request onward: the visitor's question is already in
  // the transcript by the time this runs, whether it was sent straight away or
  // held through the backend's warm-up.
  const startExchange = useCallback(
    (trimmed: string, history: ChatTurn[]) => {
      const assistantId = makeId()
      // The assistant message starts empty and fills in as tokens stream;
      // components hide assistant bubbles with no content yet.
      setMessages((prev) => [...prev, { id: assistantId, role: 'assistant', content: '' }])
      setChatStatus('retrieving')

      const controller = new AbortController()
      abortRef.current = controller

      async function run() {
        let gotContent = false
        try {
          const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: trimmed, history }),
            signal: controller.signal,
          })

          if (!response.ok || !response.body) {
            setErrorMessage(response.status === 429 ? RATE_LIMIT_ERROR : GENERIC_ERROR)
            setChatStatus('error')
            removeIfEmpty(assistantId)
            return
          }

          const parser = createStreamParser()
          const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()

          let finished = false
          const handleEvent = (event: ReturnType<typeof parser.feed>[number]) => {
            switch (event.type) {
              case 'sources':
                updateMessage(assistantId, (m) => ({
                  ...m,
                  sources: event.sources as Source[],
                }))
                setChatStatus('thinking')
                break
              case 'token':
                if (!gotContent) {
                  gotContent = true
                  setChatStatus('streaming')
                }
                updateMessage(assistantId, (m) => ({ ...m, content: m.content + event.token }))
                break
              case 'error':
                setErrorMessage(GENERIC_ERROR)
                setChatStatus('error')
                finished = true
                break
              case 'done':
                setChatStatus('idle')
                finished = true
                break
            }
          }

          while (!finished) {
            const { done, value } = await reader.read()
            if (done) {
              parser.flush().forEach(handleEvent)
              break
            }
            for (const event of parser.feed(value)) {
              handleEvent(event)
              if (finished) break
            }
          }

          if (!finished) {
            // Stream closed without a d: marker — treat as complete if we got text
            setChatStatus(gotContent ? 'idle' : 'error')
            if (!gotContent) setErrorMessage(GENERIC_ERROR)
          }
        } catch (err) {
          if ((err as Error).name === 'AbortError') {
            setChatStatus('idle')
          } else {
            setErrorMessage(GENERIC_ERROR)
            setChatStatus('error')
          }
        } finally {
          removeIfEmpty(assistantId)
          abortRef.current = null
        }
      }

      void run()
    },
    [removeIfEmpty, updateMessage]
  )

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      // A held question is work in flight: it holds the turn like a request does.
      if (chatStatus !== 'idle' && chatStatus !== 'error') return

      const history = messages
        .filter((m) => m.content.length > 0)
        .slice(-HISTORY_LIMIT)
        .map((m) => ({ role: m.role, content: m.content }))

      // The question is accepted either way — the visitor sees it in the
      // transcript, not swallowed while the backend wakes.
      setMessages((prev) => [...prev, { id: makeId(), role: 'user', content: trimmed }])
      setErrorMessage(null)

      // Hold unless the backend is known to be ready. Readiness is still
      // `null` for the first round trip after mount — exactly when a starter
      // question is clicked — and a question sent then would go into a backend
      // that may well be cold. A `failed` warm-up is held too, so the dispatch
      // effect below turns it into an error rather than an opaque failure.
      if (readiness !== 'ready') {
        heldRef.current = { text: trimmed, history }
        setChatStatus('held')
        setHeldCount((n) => n + 1)
        return
      }

      startExchange(trimmed, history)
    },
    [messages, chatStatus, readiness, startExchange]
  )

  // Dispatch a held question as soon as readiness resolves. The question is
  // claimed out of the ref before it is sent, so it goes out exactly once
  // however often readiness is observed or this effect re-runs.
  useEffect(() => {
    if (readiness !== 'ready' && readiness !== 'failed') return
    const held = heldRef.current
    if (!held) return
    heldRef.current = null

    if (readiness === 'ready') {
      startExchange(held.text, held.history)
      return
    }
    // A pipeline that failed to load will never become ready: say so rather
    // than leaving the question waiting for a wake that never comes.
    setErrorMessage(WARMUP_FAILED_ERROR)
    setChatStatus('error')
  }, [readiness, heldCount, startExchange])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    // Stop is shown while a question is held too, and it has to mean something
    // there: drop the question rather than dispatching it on the next wake.
    if (heldRef.current) {
      heldRef.current = null
      setChatStatus('idle')
    }
  }, [])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    // Drop a question still waiting on the warm-up, so a later wake does not
    // dispatch a question the visitor has cleared away.
    heldRef.current = null
    setMessages([])
    setErrorMessage(null)
    setChatStatus('idle')
  }, [])

  return { messages, status, errorMessage, sendMessage, stop, clear }
}
