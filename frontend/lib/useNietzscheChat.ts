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

let nextId = 0
function makeId(): string {
  nextId += 1
  return `msg-${nextId}`
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

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      if (chatStatus !== 'idle' && chatStatus !== 'error') return

      const history = messages
        .filter((m) => m.content.length > 0)
        .slice(-HISTORY_LIMIT)
        .map((m) => ({ role: m.role, content: m.content }))

      const userMessage: ChatMessageData = { id: makeId(), role: 'user', content: trimmed }
      const assistantId = makeId()
      // The assistant message starts empty and fills in as tokens stream;
      // components hide assistant bubbles with no content yet.
      setMessages((prev) => [
        ...prev,
        userMessage,
        { id: assistantId, role: 'assistant', content: '' },
      ])
      setErrorMessage(null)
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
    [messages, chatStatus, removeIfEmpty, updateMessage]
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setMessages([])
    setErrorMessage(null)
    setChatStatus('idle')
  }, [])

  return { messages, status, errorMessage, sendMessage, stop, clear }
}
