import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import StarterQuestions from '@/components/chat/StarterQuestions'
import { STARTER_QUESTIONS } from '@/lib/starterQuestions'

describe('StarterQuestions', () => {
  it('renders all six starter questions', () => {
    render(<StarterQuestions onSelect={vi.fn()} />)
    expect(STARTER_QUESTIONS).toHaveLength(6)
    for (const question of STARTER_QUESTIONS) {
      expect(screen.getByRole('button', { name: question })).toBeInTheDocument()
    }
  })

  it('calls onSelect with the question text when clicked', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()
    render(<StarterQuestions onSelect={onSelect} />)

    await user.click(screen.getByRole('button', { name: 'What is the will to power?' }))
    expect(onSelect).toHaveBeenCalledWith('What is the will to power?')
  })
})
