'use client'

import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import Header from '@/components/layout/Header'
import InfoPanel from '@/components/info/InfoPanel'
import ChatMessage from './ChatMessage'
import ChatInput from './ChatInput'
import StarterQuestions from './StarterQuestions'
import StatusIndicator from './StatusIndicator'
import useNietzscheChat from '@/lib/useNietzscheChat'

const SESSION_LIMIT = 50
const DEBOUNCE_MS = 3000

export default function ChatShell() {
  const { messages, status, errorMessage, sendMessage, stop, clear } = useNietzscheChat()
  const [infoOpen, setInfoOpen] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [lastSubmitTime, setLastSubmitTime] = useState(0)
  const [debounced, setDebounced] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const isLoading = status === 'retrieving' || status === 'thinking' || status === 'streaming'
  const userMessageCount = messages.filter((m) => m.role === 'user').length
  const limitReached = userMessageCount >= SESSION_LIMIT

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, status])

  useEffect(() => {
    if (!isLoading && !debounced && !limitReached) {
      inputRef.current?.focus()
    }
  }, [isLoading, debounced, limitReached])

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    }
  }, [])

  function submitQuestion(text: string) {
    if (isLoading || debounced || limitReached) return

    const now = Date.now()
    if (now - lastSubmitTime < DEBOUNCE_MS) {
      setDebounced(true)
      debounceTimerRef.current = setTimeout(
        () => setDebounced(false),
        DEBOUNCE_MS - (now - lastSubmitTime)
      )
      return
    }

    sendMessage(text)
    setInputValue('')
    setLastSubmitTime(now)

    setDebounced(true)
    debounceTimerRef.current = setTimeout(() => setDebounced(false), DEBOUNCE_MS)
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const text = inputValue.trim()
    if (text) submitQuestion(text)
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    setInputValue(e.target.value)
  }

  const lastMessage = messages[messages.length - 1]
  const streamingId =
    status === 'streaming' && lastMessage?.role === 'assistant' ? lastMessage.id : null

  return (
    <div className="flex min-h-screen flex-col">
      <Header
        onOpenInfo={() => setInfoOpen(true)}
        onClearChat={clear}
        hasMessages={messages.length > 0}
      />

      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 sm:px-6">
        {messages.length === 0 && (status === 'idle' || status === 'waking') ? (
          // A waking backend keeps the welcome screen: the cold start is meant
          // to pass while the visitor reads and types, not behind a spinner.
          // See docs/adr/0001-scale-to-zero-with-warm-on-arrival.md.
          <>
            <StarterQuestions onSelect={submitQuestion} />
            <div className="flex justify-center pb-6">
              <StatusIndicator status={status} />
            </div>
          </>
        ) : (
          <div
            className="flex flex-1 flex-col gap-4 py-6"
            aria-live="polite"
            aria-atomic="false"
            aria-relevant="additions"
          >
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                streaming={message.id === streamingId}
              />
            ))}

            <StatusIndicator status={status} />

            {status === 'error' && errorMessage && (
              <div className="flex justify-start gap-2">
                <span className="mt-1 shrink-0 font-mono text-[10px] tracking-widest text-red-400">
                  ERR
                </span>
                <div className="max-w-[85%] rounded-xl bg-red-500/5 px-4 py-3 text-[13px] text-red-400 shadow-[inset_0_0_0_1px_rgba(239,68,68,0.2)]">
                  {errorMessage}
                </div>
              </div>
            )}

            {limitReached && (
              <div className="text-center font-mono text-[11px] tracking-widest text-[#8a8f98]">
                — session limit reached —
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="sticky bottom-0 z-20 border-t border-white/[0.06] bg-[#050506]/80 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-3xl items-center gap-2 px-4 py-3 sm:px-6">
          <div className="flex-1">
            <ChatInput
              ref={inputRef}
              value={inputValue}
              onChange={handleInputChange}
              onSubmit={handleSubmit}
              disabled={isLoading}
              debounced={debounced}
              limitReached={limitReached}
            />
          </div>
          {isLoading && (
            <button
              onClick={stop}
              aria-label="Stop generating"
              className="flex h-11 shrink-0 items-center rounded-lg border border-white/[0.08] px-3 font-mono text-[11px] tracking-widest text-[#8a8f98] transition-colors hover:text-[#ededef]"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      <InfoPanel open={infoOpen} onClose={() => setInfoOpen(false)} />
    </div>
  )
}
