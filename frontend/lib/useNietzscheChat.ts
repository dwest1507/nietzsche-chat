'use client'

import { useCallback, useRef, useState } from 'react'
import { createStreamParser } from './streamParser'
import type { ChatMessageData, ChatStatus, Source } from './types'

const HISTORY_LIMIT = 10

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
  const [status, setStatus] = useState<ChatStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

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
      if (status !== 'idle' && status !== 'error') return

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
      setStatus('retrieving')

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
            setStatus('error')
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
                setStatus('thinking')
                break
              case 'token':
                if (!gotContent) {
                  gotContent = true
                  setStatus('streaming')
                }
                updateMessage(assistantId, (m) => ({ ...m, content: m.content + event.token }))
                break
              case 'error':
                setErrorMessage(GENERIC_ERROR)
                setStatus('error')
                finished = true
                break
              case 'done':
                setStatus('idle')
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
            setStatus(gotContent ? 'idle' : 'error')
            if (!gotContent) setErrorMessage(GENERIC_ERROR)
          }
        } catch (err) {
          if ((err as Error).name === 'AbortError') {
            setStatus('idle')
          } else {
            setErrorMessage(GENERIC_ERROR)
            setStatus('error')
          }
        } finally {
          removeIfEmpty(assistantId)
          abortRef.current = null
        }
      }

      void run()
    },
    [messages, status, removeIfEmpty, updateMessage]
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setMessages([])
    setErrorMessage(null)
    setStatus('idle')
  }, [])

  return { messages, status, errorMessage, sendMessage, stop, clear }
}
