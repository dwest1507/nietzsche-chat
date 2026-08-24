import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ChatMessage from '@/components/chat/ChatMessage'
import type { ChatMessageData } from '@/lib/types'

const XSS_TEXT = 'Beware <img src=x onerror=alert(1)> of <b>markup</b>'

describe('ChatMessage', () => {
  it('renders user messages with the YOU label', () => {
    const message: ChatMessageData = { id: '1', role: 'user', content: 'What is truth?' }
    render(<ChatMessage message={message} />)
    expect(screen.getByText('What is truth?')).toBeInTheDocument()
    expect(screen.getByText('YOU')).toBeInTheDocument()
  })

  it('renders assistant messages with the FN label', () => {
    const message: ChatMessageData = {
      id: '2',
      role: 'assistant',
      content: 'Truth is a mobile army of metaphors.',
    }
    render(<ChatMessage message={message} />)
    expect(screen.getByText('Truth is a mobile army of metaphors.')).toBeInTheDocument()
    expect(screen.getByText('FN')).toBeInTheDocument()
  })

  it('renders nothing for an empty in-flight assistant message', () => {
    const message: ChatMessageData = { id: '3', role: 'assistant', content: '' }
    const { container } = render(<ChatMessage message={message} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders content as plain text, not HTML', () => {
    const message: ChatMessageData = { id: '4', role: 'assistant', content: XSS_TEXT }
    const { container } = render(<ChatMessage message={message} />)
    expect(screen.getByText(XSS_TEXT)).toBeInTheDocument()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('b')).toBeNull()
  })

  it('shows source citations on completed assistant messages', () => {
    const message: ChatMessageData = {
      id: '5',
      role: 'assistant',
      content: 'So spake I.',
      sources: [
        {
          title: 'Thus Spake Zarathustra',
          translator: 'Thomas Common',
          url: 'https://www.gutenberg.org/',
          text: 'I teach you the Superman.',
        },
      ],
    }
    render(<ChatMessage message={message} />)
    expect(screen.getByText(/Source passages · 1/)).toBeInTheDocument()
  })

  it('hides source citations while streaming', () => {
    const message: ChatMessageData = {
      id: '6',
      role: 'assistant',
      content: 'So spake…',
      sources: [
        {
          title: 'Thus Spake Zarathustra',
          translator: 'Thomas Common',
          url: 'https://www.gutenberg.org/',
          text: 'I teach you the Superman.',
        },
      ],
    }
    render(<ChatMessage message={message} streaming />)
    expect(screen.queryByText(/Source passages/)).not.toBeInTheDocument()
  })
})
