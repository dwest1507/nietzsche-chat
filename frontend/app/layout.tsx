import type { Metadata } from 'next'
import { Inter, Crimson_Pro } from 'next/font/google'
import AmbientBackground from '@/components/layout/AmbientBackground'
import './globals.css'

const inter = Inter({
  variable: '--font-inter',
  subsets: ['latin'],
  display: 'swap',
})

const crimson = Crimson_Pro({
  variable: '--font-crimson',
  subsets: ['latin'],
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Chat with Friedrich Nietzsche',
  description:
    'Converse with Friedrich Nietzsche. Answers are grounded in passages retrieved from 19 of his works via hybrid semantic + keyword search.',
  openGraph: {
    title: 'Chat with Friedrich Nietzsche',
    description:
      'Converse with Friedrich Nietzsche. Answers are grounded in passages retrieved from 19 of his works.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${crimson.variable}`}>
      <body className="flex min-h-screen flex-col">
        <AmbientBackground />
        <main className="relative z-10 flex min-h-screen flex-1 flex-col">{children}</main>
      </body>
    </html>
  )
}
