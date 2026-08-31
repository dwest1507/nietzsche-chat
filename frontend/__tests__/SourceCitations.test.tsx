import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SourceCitations from '@/components/chat/SourceCitations'
import type { Source } from '@/lib/types'

const SOURCES: Source[] = [
  {
    title: 'Beyond Good and Evil',
    translator: 'Helen Zimmern',
    url: 'https://www.gutenberg.org/',
    text: 'What is done out of love always takes place beyond good and evil.',
  },
  {
    title: 'The Antichrist',
    translator: 'H. L. Mencken',
    url: '',
    text: 'Passage with <script>alert(1)</script> markup inside.',
  },
]

describe('SourceCitations', () => {
  it('starts collapsed showing only the count', () => {
    render(<SourceCitations sources={SOURCES} />)
    expect(screen.getByText(/Source passages · 2/)).toBeInTheDocument()
    expect(screen.queryByText(/beyond good and evil\./)).not.toBeInTheDocument()
    expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'false')
  })

  it('expands to show titles, translators, and passages', async () => {
    const user = userEvent.setup()
    render(<SourceCitations sources={SOURCES} />)
    await user.click(screen.getByRole('button'))

    expect(screen.getByText('Beyond Good and Evil')).toBeInTheDocument()
    expect(screen.getByText(/Helen Zimmern/)).toBeInTheDocument()
    expect(
      screen.getByText('What is done out of love always takes place beyond good and evil.')
    ).toBeInTheDocument()
  })

  it('links the title to the source url when present', async () => {
    const user = userEvent.setup()
    render(<SourceCitations sources={SOURCES} />)
    await user.click(screen.getByRole('button'))

    const link = screen.getByRole('link', { name: 'Beyond Good and Evil' })
    expect(link).toHaveAttribute('href', 'https://www.gutenberg.org/')
    // No url — plain text, no link
    expect(screen.queryByRole('link', { name: 'The Antichrist' })).not.toBeInTheDocument()
  })

  it('renders passage text as plain text, not HTML', async () => {
    const user = userEvent.setup()
    const { container } = render(<SourceCitations sources={SOURCES} />)
    await user.click(screen.getAllByRole('button')[0])

    expect(screen.getByText(/alert\(1\)/)).toBeInTheDocument()
    expect(container.querySelector('script')).toBeNull()
  })
})
