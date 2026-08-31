import { describe, it, expect } from 'vitest'
import { createStreamParser, type StreamEvent } from '@/lib/streamParser'

const SOURCES = [
  {
    title: 'Beyond Good and Evil',
    translator: 'Helen Zimmern',
    url: 'https://www.gutenberg.org/',
    text: 'What is done out of love always takes place beyond good and evil.',
  },
]

describe('createStreamParser', () => {
  it('parses token lines', () => {
    const parser = createStreamParser()
    const events = parser.feed('0:"The"\n0:" will"\n')
    expect(events).toEqual([
      { type: 'token', token: 'The' },
      { type: 'token', token: ' will' },
    ])
  })

  it('parses a sources line', () => {
    const parser = createStreamParser()
    const events = parser.feed(`2:${JSON.stringify(SOURCES)}\n`)
    expect(events).toEqual([{ type: 'sources', sources: SOURCES }])
  })

  it('parses error and done lines', () => {
    const parser = createStreamParser()
    expect(parser.feed('3:{"category":"generic"}\n')).toEqual([
      { type: 'error', category: 'generic' },
    ])
    expect(parser.feed('d:{"finishReason": "stop"}\n')).toEqual([{ type: 'done' }])
  })

  it('parses each error category the backend emits', () => {
    const parser = createStreamParser()
    expect(parser.feed('3:{"category":"provider_quota"}\n')).toEqual([
      { type: 'error', category: 'provider_quota' },
    ])
    expect(parser.feed('3:{"category":"generic"}\n')).toEqual([
      { type: 'error', category: 'generic' },
    ])
  })

  it('degrades an unknown or future error category to generic', () => {
    // A category added to the backend later must not break an older client.
    const parser = createStreamParser()
    expect(parser.feed('3:{"category":"corpus_on_fire"}\n')).toEqual([
      { type: 'error', category: 'generic' },
    ])
  })

  it('degrades an error line with no category at all to generic', () => {
    const parser = createStreamParser()
    expect(parser.feed('3:{}\n')).toEqual([{ type: 'error', category: 'generic' }])
    expect(parser.feed('3:"Generation failed"\n')).toEqual([{ type: 'error', category: 'generic' }])
  })

  it('does not break the stream on a malformed error line', () => {
    const parser = createStreamParser()
    expect(parser.feed('3:{"category":\n0:"still parsing"\n')).toEqual([
      { type: 'token', token: 'still parsing' },
    ])
  })

  it('buffers lines split across chunk boundaries', () => {
    const parser = createStreamParser()
    expect(parser.feed('0:"Übermen')).toEqual([])
    expect(parser.feed('sch"\n0:"!')).toEqual([{ type: 'token', token: 'Übermensch' }])
    expect(parser.feed('"\n')).toEqual([{ type: 'token', token: '!' }])
  })

  it('handles a full stream in one chunk', () => {
    const parser = createStreamParser()
    const raw = `2:${JSON.stringify(SOURCES)}\n0:"God"\n0:" is dead."\nd:{"finishReason": "stop"}\n`
    const types = parser.feed(raw).map((e: StreamEvent) => e.type)
    expect(types).toEqual(['sources', 'token', 'token', 'done'])
  })

  it('skips malformed lines', () => {
    const parser = createStreamParser()
    const events = parser.feed('0:not-json\nx:"unknown"\n\n0:"ok"\n')
    expect(events).toEqual([{ type: 'token', token: 'ok' }])
  })

  it('flush parses a trailing line without newline', () => {
    const parser = createStreamParser()
    expect(parser.feed('0:"tail"')).toEqual([])
    expect(parser.flush()).toEqual([{ type: 'token', token: 'tail' }])
    expect(parser.flush()).toEqual([])
  })
})
