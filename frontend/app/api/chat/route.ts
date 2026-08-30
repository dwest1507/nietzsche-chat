import type { NextRequest } from 'next/server'

import { backendFetch } from '@/lib/backendClient'

const MAX_MESSAGE_LENGTH = 1000

interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export async function POST(request: NextRequest) {
  let body: { message?: unknown; history?: unknown }
  try {
    body = await request.json()
  } catch {
    return new Response('Invalid request', { status: 400 })
  }

  const message = typeof body.message === 'string' ? body.message.trim() : ''
  if (!message) {
    return new Response('Message cannot be empty', { status: 400 })
  }
  if (message.length > MAX_MESSAGE_LENGTH) {
    return new Response(`Message too long (max ${MAX_MESSAGE_LENGTH} characters)`, { status: 400 })
  }

  const history: HistoryMessage[] = Array.isArray(body.history)
    ? body.history
        .filter(
          (m): m is HistoryMessage =>
            typeof m === 'object' &&
            m !== null &&
            ((m as HistoryMessage).role === 'user' || (m as HistoryMessage).role === 'assistant') &&
            typeof (m as HistoryMessage).content === 'string'
        )
        .map((m) => ({ role: m.role, content: m.content }))
    : []

  let backendResponse: Response
  try {
    // backendFetch owns the backend URL and the shared secret header; the
    // secret is server-side only and must never reach the response below.
    backendResponse = await backendFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history }),
      // Forward the client's disconnect so pressing Stop actually stops the
      // backend. Without it the generation runs to completion, billing Groq
      // tokens and holding a worker for an answer nobody will read.
      signal: request.signal,
    })
  } catch {
    return new Response('Service unavailable', { status: 502 })
  }

  if (!backendResponse.ok) {
    if (backendResponse.status === 429) {
      return new Response('Rate limit exceeded', { status: 429 })
    }
    if (backendResponse.status === 422 || backendResponse.status === 400) {
      return new Response('Invalid request', { status: 400 })
    }
    return new Response('Backend error', { status: 502 })
  }

  if (!backendResponse.body) {
    return new Response('Empty response from backend', { status: 502 })
  }

  // The backend speaks the data-stream line protocol the client parses
  // (lib/streamParser.ts), so pipe the body through unchanged.
  return new Response(backendResponse.body, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache',
      'X-Accel-Buffering': 'no',
    },
  })
}
