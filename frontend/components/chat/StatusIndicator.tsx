import type { ChatStatus } from '@/lib/types'

interface StatusIndicatorProps {
  status: ChatStatus
}

const STATUS_TEXT: Partial<Record<ChatStatus, string>> = {
  retrieving: 'Searching through 19 works of Nietzsche…',
  thinking: 'Nietzsche is contemplating…',
}

export default function StatusIndicator({ status }: StatusIndicatorProps) {
  const text = STATUS_TEXT[status]
  if (!text) return null

  return (
    <div className="flex gap-2">
      <span className="mt-1 shrink-0 font-mono text-[10px] tracking-widest text-[#0ea5e9]">FN</span>
      <div
        role="status"
        className="rounded-xl bg-white/[0.05] px-4 py-3 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.07)]"
      >
        <span className="animate-[pulse-soft_1.6s_ease-in-out_infinite] text-[13px] text-[#8a8f98] italic">
          {text}
        </span>
      </div>
    </div>
  )
}
