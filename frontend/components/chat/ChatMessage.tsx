import { memo } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import SourceCitations from './SourceCitations'
import type { ChatMessageData } from '@/lib/types'

interface ChatMessageProps {
  message: ChatMessageData
  streaming?: boolean
}

// Defined once at module scope: re-creating this object on every render would
// make react-markdown re-parse the whole answer on every streamed token.
const markdownComponents: Components = {
  // `whitespace-pre-line` keeps the single newlines the model writes between
  // lines. Markdown itself folds them into spaces, which turned every stanza
  // and short-line list into one run-on paragraph.
  p: ({ children }) => <p className="mb-3 whitespace-pre-line last:mb-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
  em: ({ children }) => <em className="text-[#ededef] italic">{children}</em>,
  blockquote: ({ children }) => (
    <blockquote className="my-2.5 rounded-r-md border-l-2 border-[#0ea5e9]/60 bg-white/[0.03] py-1.5 pr-2 pl-3.5 font-serif text-[14.5px] text-[#ededef]/90 italic">
      {children}
    </blockquote>
  ),
  ul: ({ children }) => <ul className="my-2 ml-4 list-disc space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 ml-4 list-decimal space-y-1">{children}</ol>,
  li: ({ children }) => <li className="pl-0.5 leading-relaxed">{children}</li>,
  h1: ({ children }) => (
    <h1 className="mt-3.5 mb-2 font-serif text-lg font-semibold text-white first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-3 mb-1.5 font-serif text-base font-semibold text-white first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-2.5 mb-1 font-serif text-[15px] font-medium text-white first:mt-0">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-2 mb-1 text-[14px] font-medium text-white first:mt-0">{children}</h4>
  ),
  hr: () => <hr className="my-3 border-white/[0.08]" />,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-[#38bdf8] underline underline-offset-2 transition-colors hover:text-[#0ea5e9]"
    >
      {children}
    </a>
  ),
  // Answers are generated from retrieved corpus text, so an `![](...)` in the
  // output would make the reader's browser fetch from an arbitrary host —
  // leaking their IP and user agent. Nothing here needs images: show the alt
  // text instead.
  img: ({ alt }) => (alt ? <span className="text-[#8a8f98] italic">{alt}</span> : null),
  code: ({ node, className, children, ...props }) => {
    void node // react-markdown passes the AST node; never spread it onto the DOM
    const isInline = !className && typeof children === 'string' && !children.includes('\n')
    if (isInline) {
      return (
        <code
          className="rounded bg-white/10 px-1 py-0.5 font-mono text-[13px] text-[#38bdf8]"
          {...props}
        >
          {children}
        </code>
      )
    }
    return (
      <code className={className} {...props}>
        {children}
      </code>
    )
  },
  pre: ({ children }) => (
    <pre className="my-2.5 overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3 font-mono text-xs text-[#38bdf8]">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="my-2.5 overflow-x-auto">
      <table className="min-w-full divide-y divide-white/10 text-left text-[13px]">
        {children}
      </table>
    </div>
  ),
  th: ({ children }) => (
    <th className="bg-white/[0.04] px-3 py-1.5 font-medium text-white">{children}</th>
  ),
  td: ({ children }) => <td className="border-t border-white/[0.06] px-3 py-1.5">{children}</td>,
}

const remarkPlugins = [remarkGfm]

function ChatMessage({ message, streaming = false }: ChatMessageProps) {
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
          {isUser ? (
            <span className="break-words whitespace-pre-wrap">{message.content}</span>
          ) : (
            // The cursor span, not the last paragraph, is this div's last child,
            // so the caret has to key off `p:last-of-type` to sit on the same
            // line as the text it trails.
            <div
              className={`text-[14px] leading-relaxed break-words ${streaming ? '[&>p:last-of-type]:inline' : ''}`}
            >
              <ReactMarkdown remarkPlugins={remarkPlugins} components={markdownComponents}>
                {message.content}
              </ReactMarkdown>
              {streaming && (
                <span
                  aria-hidden="true"
                  data-testid="streaming-cursor"
                  className="ml-0.5 inline-block h-4 w-[2px] translate-y-[2px] animate-[blink_1s_step-end_infinite] bg-[#38bdf8]"
                />
              )}
            </div>
          )}
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

// Every streamed token re-renders the whole thread; without this each already
// finished message re-parses its markdown on each one.
export default memo(ChatMessage)
