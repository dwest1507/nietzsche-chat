import type { Source } from './types'

/**
 * Incremental parser for the backend's AI SDK v1 data-stream line protocol:
 *   2:[{...sources}]              source passages (sent first)
 *   0:"token"                     one generated text token
 *   3:{"category":"generic"}      error, as a category — never provider text
 *   d:{"finishReason":"stop"}     end of stream
 *
 * Feed it decoded text chunks as they arrive; it buffers partial lines across
 * chunk boundaries and skips malformed lines.
 */

/**
 * Why the backend failed, as far as the visitor needs to know. The backend may
 * grow categories this client has never heard of, so anything unrecognised —
 * including the older bare-string error line — resolves to 'generic' rather
 * than breaking the stream. See backend/app/routes/chat.py.
 *
 * The per-visitor rate limit is deliberately absent: it is rejected with HTTP
 * 429 before the stream starts, so it can never arrive as a category.
 */
export type StreamErrorCategory = 'provider_quota' | 'generic'

const ERROR_CATEGORIES: readonly string[] = ['provider_quota', 'generic']

export type StreamEvent =
  | { type: 'sources'; sources: Source[] }
  | { type: 'token'; token: string }
  | { type: 'error'; category: StreamErrorCategory }
  | { type: 'done' }

function errorCategory(payload: unknown): StreamErrorCategory {
  const category =
    typeof payload === 'object' && payload !== null
      ? (payload as { category?: unknown }).category
      : undefined
  return typeof category === 'string' && ERROR_CATEGORIES.includes(category)
    ? (category as StreamErrorCategory)
    : 'generic'
}

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
      case '3:':
        return { type: 'error', category: errorCategory(JSON.parse(payload)) }
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
