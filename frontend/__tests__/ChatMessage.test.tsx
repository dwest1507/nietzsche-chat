import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ChatMessage from '@/components/chat/ChatMessage'
import type { ChatMessageData } from '@/lib/types'

const XSS_TEXT = 'Beware <script>alert("xss")</script><img src=x onerror=alert(1)> of <b>markup</b>'

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

  it('safely escapes raw HTML and prevents script/image tag injection', () => {
    const message: ChatMessageData = { id: '4', role: 'assistant', content: XSS_TEXT }
    const { container } = render(<ChatMessage message={message} />)
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })

  it('renders bold, italic, and bold-italic markdown formatting', () => {
    const markdownContent =
      '***“God is dead.”*** The belief has become **unworthy of belief** and *trembles*.'
    const message: ChatMessageData = { id: '5', role: 'assistant', content: markdownContent }
    const { container } = render(<ChatMessage message={message} />)

    const strongElements = container.querySelectorAll('strong')
    expect(strongElements.length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('“God is dead.”')).toBeInTheDocument()
    expect(screen.getByText('unworthy of belief')).toBeInTheDocument()

    const emElements = container.querySelectorAll('em')
    expect(emElements.length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('trembles')).toBeInTheDocument()
  })

  it('renders markdown blockquotes', () => {
    const markdownContent = '> *“We have killed Him—you and I! ...”* (Passage 125).'
    const message: ChatMessageData = { id: '6', role: 'assistant', content: markdownContent }
    const { container } = render(<ChatMessage message={message} />)

    const blockquote = container.querySelector('blockquote')
    expect(blockquote).not.toBeNull()
    expect(blockquote?.textContent).toContain('We have killed Him—you and I!')
  })

  it('renders ordered and unordered lists', () => {
    const markdownContent = `Thus, "God is dead" means:

1. **The death of the old metaphysics**
2. **The collapse of the moral edifice**

Key concepts:
- Will to Power
- Eternal Recurrence`

    const message: ChatMessageData = { id: '7', role: 'assistant', content: markdownContent }
    const { container } = render(<ChatMessage message={message} />)

    const ol = container.querySelector('ol')
    expect(ol).not.toBeNull()
    const olItems = ol?.querySelectorAll('li')
    expect(olItems?.length).toBe(2)
    expect(screen.getByText('The death of the old metaphysics')).toBeInTheDocument()

    const ul = container.querySelector('ul')
    expect(ul).not.toBeNull()
    const ulItems = ul?.querySelectorAll('li')
    expect(ulItems?.length).toBe(2)
    expect(screen.getByText('Will to Power')).toBeInTheDocument()
    expect(screen.getByText('Eternal Recurrence')).toBeInTheDocument()
  })

  it('renders headings, inline code, code blocks, and external links', () => {
    const markdownContent = `# The Übermensch
## Higher Man
Use \`console.log\` to inspect:
\`\`\`js
const spirit = "camel"
\`\`\`
Read more on [Gutenberg](https://www.gutenberg.org/).`

    const message: ChatMessageData = { id: '8', role: 'assistant', content: markdownContent }
    const { container } = render(<ChatMessage message={message} />)

    expect(container.querySelector('h1')?.textContent).toBe('The Übermensch')
    expect(container.querySelector('h2')?.textContent).toBe('Higher Man')
    expect(container.querySelector('code')?.textContent).toBe('console.log')
    expect(container.querySelector('pre')).not.toBeNull()

    const link = container.querySelector('a')
    expect(link).not.toBeNull()
    expect(link?.getAttribute('href')).toBe('https://www.gutenberg.org/')
    expect(link?.getAttribute('target')).toBe('_blank')
    expect(link?.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('renders streaming cursor during active streaming', () => {
    const message: ChatMessageData = {
      id: '9',
      role: 'assistant',
      content: 'Streaming in progress…',
    }
    const { rerender } = render(<ChatMessage message={message} streaming={true} />)

    expect(screen.getByTestId('streaming-cursor')).toBeInTheDocument()

    rerender(<ChatMessage message={message} streaming={false} />)
    expect(screen.queryByTestId('streaming-cursor')).not.toBeInTheDocument()
  })

  it('preserves the single newlines the model writes inside a paragraph', () => {
    const message: ChatMessageData = {
      id: '10',
      role: 'assistant',
      content: 'Line one\nLine two\nLine three',
    }
    const { container } = render(<ChatMessage message={message} />)
    const paragraph = container.querySelector('p')
    // Markdown folds a single newline into a space, so the line breaks only
    // survive if the paragraph opts into pre-line whitespace handling.
    expect(paragraph?.textContent).toBe('Line one\nLine two\nLine three')
    expect(paragraph?.className).toContain('whitespace-pre-line')
  })

  it('does not fetch images referenced by markdown in the answer', () => {
    const message: ChatMessageData = {
      id: '11',
      role: 'assistant',
      content: 'Consider this ![a tracking pixel](https://evil.example/p.png) closely.',
    }
    const { container } = render(<ChatMessage message={message} />)
    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText('a tracking pixel')).toBeInTheDocument()
  })

  it('does not leak the react-markdown AST node onto rendered elements', () => {
    const message: ChatMessageData = { id: '12', role: 'assistant', content: 'Use `amor fati`.' }
    const { container } = render(<ChatMessage message={message} />)
    expect(container.querySelector('code')?.hasAttribute('node')).toBe(false)
  })

  it('places the streaming cursor after the final paragraph', () => {
    const message: ChatMessageData = {
      id: '13',
      role: 'assistant',
      content: 'First thought.\n\nSecond thought',
    }
    const { container } = render(<ChatMessage message={message} streaming />)
    const wrapper = screen.getByTestId('streaming-cursor').parentElement
    // The cursor span is the wrapper's last child, so a `p:last-child` rule
    // would match nothing and drop the caret onto its own line.
    expect(wrapper?.className).toContain('[&>p:last-of-type]:inline')
    expect(container.querySelector('[class*="last-of-type"] > p:last-of-type')?.textContent).toBe(
      'Second thought'
    )
  })

  it('shows source citations on completed assistant messages', () => {
    const message: ChatMessageData = {
      id: '10',
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
      id: '11',
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
