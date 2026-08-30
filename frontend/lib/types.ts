export interface Source {
  title: string
  translator: string
  url: string
  text: string
}

export interface ChatMessageData {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

export type ChatStatus = 'idle' | 'waking' | 'retrieving' | 'thinking' | 'streaming' | 'error'

/** What the backend says about its retrieval pipeline: see `/api/ready`. */
export type ReadinessState = 'loading' | 'ready' | 'failed'
