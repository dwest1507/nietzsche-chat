'use client'

import { useState } from 'react'
import type { Source } from '@/lib/types'

interface SourceCitationsProps {
  sources: Source[]
}

export default function SourceCitations({ sources }: SourceCitationsProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="w-full">
      <button
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        className="flex min-h-[32px] items-center gap-1.5 rounded-md px-1 font-mono text-[11px] tracking-widest text-[#8a8f98] transition-colors hover:text-[#ededef]"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          className={`h-3 w-3 transition-transform ${expanded ? 'rotate-90' : ''}`}
        >
          <path d="m9 18 6-6-6-6" />
        </svg>
        Source passages · {sources.length}
      </button>

      {expanded && (
        <ul className="mt-2 flex flex-col gap-2">
          {sources.map((source, i) => (
            <li
              key={i}
              className="rounded-lg border-l-2 border-[#0ea5e9]/60 bg-white/[0.03] px-3 py-2 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.05)]"
            >
              <p className="text-[12px] font-medium text-[#38bdf8]">
                {source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline"
                  >
                    {source.title}
                  </a>
                ) : (
                  source.title
                )}
                {source.translator && (
                  <span className="ml-2 font-normal text-[#8a8f98]">
                    trans. {source.translator}
                  </span>
                )}
              </p>
              <p className="mt-1 text-[12px] leading-relaxed break-words whitespace-pre-wrap text-[#8a8f98]">
                {source.text}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
