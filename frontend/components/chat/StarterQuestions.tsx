import { STARTER_QUESTIONS } from '@/lib/starterQuestions'

interface StarterQuestionsProps {
  onSelect: (question: string) => void
}

export default function StarterQuestions({ onSelect }: StarterQuestionsProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 py-12">
      <div className="text-center">
        <p className="font-serif text-2xl text-[#ededef] sm:text-3xl">
          What would you ask Nietzsche?
        </p>
        <p className="mx-auto mt-3 max-w-md text-[13px] leading-relaxed text-[#8a8f98]">
          Every answer is grounded in passages retrieved from 19 of his works — from{' '}
          <em>Thus Spake Zarathustra</em> to <em>The Will to Power</em>.
        </p>
      </div>

      <div className="grid w-full gap-3 sm:grid-cols-2">
        {STARTER_QUESTIONS.map((question) => (
          <button
            key={question}
            onClick={() => onSelect(question)}
            className="min-h-[44px] rounded-2xl border border-white/[0.06] bg-white/[0.03] px-4 py-3 text-left text-[13px] leading-relaxed text-[#ededef] transition-all duration-150 hover:border-[#0ea5e9]/30 hover:bg-white/[0.06] active:scale-[0.99]"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}
