import type { NextRequest } from 'next/server'

import { backendFetch } from '@/lib/backendClient'

const MAX_MESSAGE_LENGTH = 1000

// The backend meters each visitor separately, but it only ever sees this
// route handler's egress address, so we hand it the visitor's. It is trusted
// there only after the shared secret checks out — which is why we must derive
// it from the platform's own forwarding headers and never pass through an
// `X-Client-IP` the browser supplied. See docs/adr/0002-shared-secret-gateway.md.
const CLIENT_ADDRESS_HEADER = 'X-Client-IP'

function visitorAddress(request: NextRequest): string | null {
  // `x-real-ip` first: the platform sets it from the connecting socket, so the
  // visitor cannot write it. `x-forwarded-for` is only as trustworthy as the
  // proxy in front — one that appends rather than replaces leaves the browser's
  // own value at the head of the chain, and since the backend meters solely on
  // the address we forward, a visitor rotating that header would get a fresh
  // bucket per request and walk past the cap protecting the Groq quota.
  const real = request.headers.get('x-real-ip')?.trim()
  if (real) return real

  // No `x-real-ip`: fall back to the chain's first entry — client, then each
  // proxy that added itself, so the visitor is the first, not the last.
  return request.headers.get('x-forwarded-for')?.split(',')[0].trim() || null
}

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

  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const address = visitorAddress(request)
  if (address) {
    headers[CLIENT_ADDRESS_HEADER] = address
  }

  let backendResponse: Response
  try {
    // backendFetch owns the backend URL and the shared secret header; the
    // secret is server-side only and must never reach the response below.
    backendResponse = await backendFetch('/api/chat', {
      method: 'POST',
      headers,
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
