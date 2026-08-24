import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatInput from '@/components/chat/ChatInput'

function renderInput(overrides: Partial<Parameters<typeof ChatInput>[0]> = {}) {
  const props = {
    value: '',
    onChange: vi.fn(),
    onSubmit: vi.fn((e: React.FormEvent) => e.preventDefault()),
    disabled: false,
    debounced: false,
    limitReached: false,
    ...overrides,
  }
  render(<ChatInput {...props} />)
  return props
}

describe('ChatInput', () => {
  it('renders the Nietzsche placeholder', () => {
    renderInput()
    expect(
      screen.getByPlaceholderText(/Ask Nietzsche about philosophy, morality/)
    ).toBeInTheDocument()
  })

  it('disables input and send button while loading', () => {
    renderInput({ disabled: true, value: 'question' })
    expect(screen.getByLabelText('Chat message input')).toBeDisabled()
    expect(screen.getByLabelText('Send message')).toBeDisabled()
  })

  it('disables the send button when the input is empty', () => {
    renderInput({ value: '   ' })
    expect(screen.getByLabelText('Send message')).toBeDisabled()
  })

  it('shows the limit placeholder when the session limit is reached', () => {
    renderInput({ limitReached: true })
    expect(screen.getByPlaceholderText(/Session limit reached/)).toBeInTheDocument()
  })

  it('submits the form on Enter', async () => {
    const user = userEvent.setup()
    const props = renderInput({ value: 'What is truth?' })
    await user.type(screen.getByLabelText('Chat message input'), '{Enter}')
    expect(props.onSubmit).toHaveBeenCalled()
  })

  it('caps input at 1000 characters', () => {
    renderInput()
    expect(screen.getByLabelText('Chat message input')).toHaveAttribute('maxLength', '1000')
  })
})
