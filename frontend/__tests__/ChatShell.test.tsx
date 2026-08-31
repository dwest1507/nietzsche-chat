import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatShell from '@/components/chat/ChatShell'
import type { NietzscheChat } from '@/lib/useNietzscheChat'
import type { ChatMessageData, ChatStatus } from '@/lib/types'

const mockChat: NietzscheChat = {
  messages: [],
  status: 'idle',
  errorMessage: null,
  sendMessage: vi.fn(),
  stop: vi.fn(),
  clear: vi.fn(),
}

vi.mock('@/lib/useNietzscheChat', () => ({
  default: () => mockChat,
}))

function setChatState(state: {
  messages?: ChatMessageData[]
  status?: ChatStatus
  errorMessage?: string | null
}) {
  mockChat.messages = state.messages ?? []
  mockChat.status = state.status ?? 'idle'
  mockChat.errorMessage = state.errorMessage ?? null
}

const CONVERSATION: ChatMessageData[] = [
  { id: '1', role: 'user', content: 'What is the will to power?' },
  { id: '2', role: 'assistant', content: 'The world is the will to power!' },
]

describe('ChatShell', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setChatState({})
  })

  it('shows the welcome screen with starter questions when empty', () => {
    render(<ChatShell />)
    expect(screen.getByText('What would you ask Nietzsche?')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'What is the will to power?' })).toBeInTheDocument()
  })

  it('sends the starter question when clicked', async () => {
    const user = userEvent.setup()
    render(<ChatShell />)
    await user.click(screen.getByRole('button', { name: 'What is the Übermensch?' }))
    expect(mockChat.sendMessage).toHaveBeenCalledWith('What is the Übermensch?')
  })

  it('renders the header title and subtitle', () => {
    render(<ChatShell />)
    expect(screen.getByText('Chat with Friedrich Nietzsche')).toBeInTheDocument()
    expect(
      screen.getByText('Thus spake Zarathustra... and now he answers your questions.')
    ).toBeInTheDocument()
  })

  it('shows the retrieval status stage', () => {
    setChatState({
      messages: [{ id: '1', role: 'user', content: 'Q' }],
      status: 'retrieving',
    })
    render(<ChatShell />)
    expect(screen.getByText('Searching through 19 works of Nietzsche…')).toBeInTheDocument()
  })

  it('shows the contemplating status stage', () => {
    setChatState({
      messages: [{ id: '1', role: 'user', content: 'Q' }],
      status: 'thinking',
    })
    render(<ChatShell />)
    expect(screen.getByText('Nietzsche is contemplating…')).toBeInTheDocument()
  })

  it('hides the status indicator and shows a stop button while streaming', () => {
    setChatState({
      messages: [
        { id: '1', role: 'user', content: 'Q' },
        { id: '2', role: 'assistant', content: 'The world…' },
      ],
      status: 'streaming',
    })
    render(<ChatShell />)
    expect(screen.queryByText(/Searching through/)).not.toBeInTheDocument()
    expect(screen.queryByText(/contemplating/)).not.toBeInTheDocument()
    expect(screen.getByLabelText('Stop generating')).toBeInTheDocument()
  })

  it('renders the conversation', () => {
    setChatState({ messages: CONVERSATION })
    render(<ChatShell />)
    expect(screen.getByText('What is the will to power?')).toBeInTheDocument()
    expect(screen.getByText('The world is the will to power!')).toBeInTheDocument()
  })

  it('clears the chat via the header button', async () => {
    const user = userEvent.setup()
    setChatState({ messages: CONVERSATION })
    render(<ChatShell />)
    await user.click(screen.getByRole('button', { name: 'Clear chat' }))
    expect(mockChat.clear).toHaveBeenCalled()
  })

  it('shows the error bubble', () => {
    setChatState({
      messages: CONVERSATION,
      status: 'error',
      errorMessage: 'Nietzsche is unreachable at the moment.',
    })
    render(<ChatShell />)
    expect(screen.getByText('Nietzsche is unreachable at the moment.')).toBeInTheDocument()
  })

  it('opens the About panel', async () => {
    const user = userEvent.setup()
    render(<ChatShell />)
    await user.click(screen.getByRole('button', { name: 'About' }))
    expect(screen.getByRole('dialog', { name: 'About this app' })).toBeVisible()
    expect(screen.getByText(/Hybrid retrieval scores every passage/)).toBeInTheDocument()
  })

  it('shows the waking notice while the backend is still stirring', () => {
    setChatState({
      messages: [{ id: '1', role: 'user', content: 'Q' }],
      status: 'waking',
    })
    render(<ChatShell />)
    expect(screen.getByText('Nietzsche is stirring…')).toBeInTheDocument()
  })

  it('keeps the starter questions on screen while the backend wakes', () => {
    // The wake is hidden behind the reading and typing, not behind a spinner:
    // see docs/adr/0001-scale-to-zero-with-warm-on-arrival.md.
    setChatState({ status: 'waking' })
    render(<ChatShell />)
    expect(screen.getByText('What would you ask Nietzsche?')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'What is the will to power?' })).toBeInTheDocument()
    expect(screen.getByText('Nietzsche is stirring…')).toBeInTheDocument()
  })

  it('leaves the input usable while the backend wakes', () => {
    setChatState({ status: 'waking' })
    render(<ChatShell />)
    expect(screen.getByLabelText('Chat message input')).toBeEnabled()
  })

  it('does not accept a second submission while a message is held', async () => {
    // A held question is work in flight: the visitor sees it accepted and the
    // backend still stirring, and cannot pile a second question on top of it.
    const user = userEvent.setup()
    setChatState({
      messages: [{ id: '1', role: 'user', content: 'What is the Übermensch?' }],
      status: 'held',
    })
    render(<ChatShell />)

    expect(screen.getByText('What is the Übermensch?')).toBeInTheDocument()
    expect(screen.getByText('Nietzsche is stirring…')).toBeInTheDocument()

    const input = screen.getByLabelText('Chat message input')
    expect(input).toBeDisabled()
    await user.type(input, 'And the eternal recurrence?{Enter}')
    expect(mockChat.sendMessage).not.toHaveBeenCalled()
  })

  it('sends a typed message through the input', async () => {
    const user = userEvent.setup()
    setChatState({ messages: CONVERSATION })
    render(<ChatShell />)
    await user.type(screen.getByLabelText('Chat message input'), 'Why?{Enter}')
    expect(mockChat.sendMessage).toHaveBeenCalledWith('Why?')
  })
})
