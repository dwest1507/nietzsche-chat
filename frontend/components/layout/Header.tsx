interface HeaderProps {
  onOpenInfo: () => void
  onClearChat: () => void
  hasMessages: boolean
}

export default function Header({ onOpenInfo, onClearChat, hasMessages }: HeaderProps) {
  return (
    <header className="sticky top-0 z-20 border-b border-white/[0.06] bg-[#050506]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-3xl items-center gap-4 px-4 py-3 sm:px-6">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#0ea5e9]/15 font-serif text-base text-[#38bdf8]">
          N
        </div>
        <div className="min-w-0 flex-1">
          <h1 className="truncate font-serif text-base font-semibold tracking-wide text-[#ededef] sm:text-lg">
            Chat with Friedrich Nietzsche
          </h1>
          <p className="hidden truncate text-[11px] text-[#8a8f98] italic sm:block">
            Thus spake Zarathustra... and now he answers your questions.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {hasMessages && (
            <button
              onClick={onClearChat}
              className="flex min-h-[44px] items-center rounded-lg border border-white/[0.08] px-3 font-mono text-[11px] tracking-widest text-[#8a8f98] transition-colors hover:border-white/[0.14] hover:text-[#ededef]"
            >
              Clear chat
            </button>
          )}
          <button
            onClick={onOpenInfo}
            aria-haspopup="dialog"
            className="flex min-h-[44px] items-center rounded-lg border border-white/[0.08] px-3 font-mono text-[11px] tracking-widest text-[#8a8f98] transition-colors hover:border-white/[0.14] hover:text-[#ededef]"
          >
            About
          </button>
        </div>
      </div>
    </header>
  )
}
