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

export type ChatStatus = 'idle' | 'retrieving' | 'thinking' | 'streaming' | 'error'
