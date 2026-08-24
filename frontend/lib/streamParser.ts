import type { Source } from './types'

/**
 * Incremental parser for the backend's AI SDK v1 data-stream line protocol:
 *   2:[{...sources}]           source passages (sent first)
 *   0:"token"                  one generated text token
 *   3:"message"                error
 *   d:{"finishReason":"stop"}  end of stream
 *
 * Feed it decoded text chunks as they arrive; it buffers partial lines across
 * chunk boundaries and skips malformed lines.
 */

export type StreamEvent =
  | { type: 'sources'; sources: Source[] }
  | { type: 'token'; token: string }
  | { type: 'error'; message: string }
  | { type: 'done' }

function parseLine(line: string): StreamEvent | null {
  if (line.length === 0) return null
  const prefix = line.slice(0, 2)
  const payload = line.slice(2)
  try {
    switch (prefix) {
      case '0:': {
        const token = JSON.parse(payload)
        return typeof token === 'string' ? { type: 'token', token } : null
      }
      case '2:': {
        const sources = JSON.parse(payload)
        return Array.isArray(sources) ? { type: 'sources', sources } : null
      }
      case '3:': {
        const message = JSON.parse(payload)
        return { type: 'error', message: typeof message === 'string' ? message : 'error' }
      }
      case 'd:':
        return { type: 'done' }
      default:
        return null
    }
  } catch {
    return null // malformed line
  }
}

export function createStreamParser() {
  let buffer = ''

  return {
    /** Parse a decoded text chunk, returning any complete events. */
    feed(chunk: string): StreamEvent[] {
      buffer += chunk
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      return lines.map(parseLine).filter((e): e is StreamEvent => e !== null)
    },

    /** Parse whatever remains in the buffer once the stream closes. */
    flush(): StreamEvent[] {
      const line = buffer
      buffer = ''
      const event = parseLine(line)
      return event ? [event] : []
    },
  }
}
