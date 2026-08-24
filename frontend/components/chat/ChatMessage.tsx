import SourceCitations from './SourceCitations'
import type { ChatMessageData } from '@/lib/types'

interface ChatMessageProps {
  message: ChatMessageData
  streaming?: boolean
}

export default function ChatMessage({ message, streaming = false }: ChatMessageProps) {
  const isUser = message.role === 'user'

  if (!message.content) return null

  return (
    <div className={`flex gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <span className="mt-1 shrink-0 font-mono text-[10px] tracking-widest text-[#0ea5e9]">
          FN
        </span>
      )}

      <div className={`flex max-w-[85%] flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-xl px-4 py-3 text-[14px] leading-relaxed ${
            isUser
              ? 'bg-[#0ea5e9]/15 text-[#ededef] shadow-[inset_0_0_0_1px_rgba(14,165,233,0.25)]'
              : 'bg-white/[0.05] text-[#ededef] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.07)]'
          }`}
        >
          <span className="break-words whitespace-pre-wrap">
            {message.content}
            {streaming && (
              <span
                aria-hidden="true"
                className="ml-0.5 inline-block h-4 w-[2px] translate-y-[2px] animate-[blink_1s_step-end_infinite] bg-[#38bdf8]"
              />
            )}
          </span>
        </div>

        {!isUser && !streaming && message.sources && message.sources.length > 0 && (
          <SourceCitations sources={message.sources} />
        )}
      </div>

      {isUser && (
        <span className="mt-1 shrink-0 font-mono text-[10px] tracking-widest text-[#8a8f98]">
          YOU
        </span>
      )}
    </div>
  )
}
