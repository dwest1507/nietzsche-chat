'use client'

import { useEffect, useRef } from 'react'

interface InfoPanelProps {
  open: boolean
  onClose: () => void
}

const WORKS = [
  'Thus Spake Zarathustra',
  'Beyond Good and Evil',
  'The Genealogy of Morals',
  'The Antichrist',
  'Ecce Homo',
  'The Twilight of the Idols',
  'The Birth of Tragedy',
  'The Joyful Wisdom',
  'The Dawn of Day',
  'Human, All-Too-Human',
  'The Will to Power (I–IV)',
  'The Case of Wagner',
  'Early Greek Philosophy',
  'Thoughts out of Season (I–II)',
  'Homer and Classical Philology',
  'On the Future of Our Educational Institutions',
  'We Philologists',
]

export default function InfoPanel({ open, onClose }: InfoPanelProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (open) closeButtonRef.current?.focus()
  }, [open])

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    if (open) {
      document.addEventListener('keydown', handleKey)
      return () => document.removeEventListener('keydown', handleKey)
    }
  }, [open, onClose])

  return (
    <>
      {/* Backdrop */}
      <div
        aria-hidden={!open}
        onClick={onClose}
        className={`fixed inset-0 z-30 bg-black/50 backdrop-blur-sm transition-opacity duration-200 ${
          open ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'
        }`}
      />

      {/* Slide-over panel */}
      <div
        role="dialog"
        aria-label="About this app"
        aria-hidden={!open}
        inert={!open}
        className={`fixed top-0 right-0 z-40 flex h-full w-full max-w-md flex-col overflow-hidden border-l border-white/[0.06] bg-[#0a0a0c]/95 shadow-[0_0_0_1px_rgba(255,255,255,0.06),0_8px_40px_rgba(0,0,0,0.6)] backdrop-blur-xl transition-transform duration-200 ${
          open ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-white/[0.06] px-5 py-4">
          <p className="font-serif text-lg text-[#ededef]">About</p>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            aria-label="Close panel"
            className="flex h-11 w-11 items-center justify-center rounded-lg text-[#8a8f98] transition-colors hover:text-[#ededef]"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="h-4 w-4"
              aria-hidden="true"
            >
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-5 text-[13px] leading-relaxed text-[#8a8f98]">
          <section>
            <h2 className="font-mono text-[11px] tracking-widest text-[#0ea5e9] uppercase">
              The philosopher answers
            </h2>
            <p className="mt-2">
              This chatbot embodies Friedrich Nietzsche using retrieval-augmented generation. Every
              response is grounded in passages retrieved from 19 of his works — public domain
              translations from Project Gutenberg — so he answers with his own words in hand, not
              from thin air.
            </p>
          </section>

          <section className="mt-6">
            <h2 className="font-mono text-[11px] tracking-widest text-[#0ea5e9] uppercase">
              Tips for great conversations
            </h2>
            <ul className="mt-2 list-disc space-y-1.5 pl-4">
              <li>
                Ask about his core ideas: the will to power, the Übermensch, eternal recurrence.
              </li>
              <li>Challenge him on morality, religion, or the meaning of suffering.</li>
              <li>Follow up — he remembers the conversation and answers in context.</li>
              <li>
                Expand &ldquo;Source passages&rdquo; under an answer to read the original text.
              </li>
            </ul>
          </section>

          <section className="mt-6">
            <h2 className="font-mono text-[11px] tracking-widest text-[#0ea5e9] uppercase">
              How it works
            </h2>
            <ol className="mt-2 list-decimal space-y-1.5 pl-4">
              <li>Follow-up questions are rewritten into standalone questions for search.</li>
              <li>
                Hybrid retrieval scores every passage: 70% semantic similarity (FAISS) + 30% keyword
                match (BM25).
              </li>
              <li>A cross-encoder re-ranks the candidates and keeps the best six passages.</li>
              <li>
                Those passages, the conversation, and Nietzsche&apos;s persona prompt are sent to a
                large language model, which streams the answer token by token.
              </li>
            </ol>
          </section>

          <section className="mt-6">
            <h2 className="font-mono text-[11px] tracking-widest text-[#0ea5e9] uppercase">
              The corpus — 19 works
            </h2>
            <ul className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2">
              {WORKS.map((work) => (
                <li key={work} className="truncate">
                  {work}
                </li>
              ))}
            </ul>
          </section>

          <p className="mt-8 border-t border-white/[0.06] pt-4 text-[11px]">
            Texts from Project Gutenberg. Built with Next.js, FastAPI, FAISS, and Groq.
          </p>
        </div>
      </div>
    </>
  )
}
