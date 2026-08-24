# Chat with Friedrich Nietzsche

A full-stack conversational AI that embodies Friedrich Nietzsche's philosophical
voice using Retrieval-Augmented Generation (RAG). Every answer is grounded in
passages retrieved from 19 of his complete works, with the source passages shown
alongside each response.

*Thus spake Zarathustra... and now he answers your questions.*

## Architecture

```
frontend/   Next.js (App Router, TypeScript, Tailwind v4)   → Vercel
backend/    FastAPI + hybrid RAG pipeline (Python, uv)      → Railway
content/    19 preprocessed Project Gutenberg works + source metadata
scripts/    CI shell scripts (single source of truth for the checks)
```

The browser talks only to the Next.js app. Its `/api/chat` route validates the
request and proxies it to the FastAPI backend (`CHAT_API_URL`, server-side
only), piping the token stream back unchanged.

### Request flow

1. **Condense** — with conversation history present, one LLM call rewrites the
   follow-up into a standalone question (used for retrieval only).
2. **Hybrid retrieval** — FAISS semantic search (`all-mpnet-base-v2`, cosine)
   and BM25 keyword search are combined 70/30 over ~7,300 chunks.
3. **Filter + re-rank** — fragments under two sentences are dropped, then a
   cross-encoder (`ms-marco-MiniLM-L-6-v2`) keeps the best 6 passages.
4. **Generate** — the Nietzsche persona prompt, the passages, the last 10
   history turns, and the question go to Groq; tokens stream straight to the
   browser.

### Stream protocol

`POST /api/chat` with `{"message": str, "history": [{role, content}]}` returns
a line-delimited stream:

```
2:[{"title", "translator", "url", "text"}, ...]   source passages (first)
0:"token"                                          one line per token
d:{"finishReason": "stop"}                         end of stream
3:"Generation failed"                              error (replaces d:)
```

## Development

Prerequisites: [uv](https://docs.astral.sh/uv/), Node.js (see
`frontend/.nvmrc`), and a free [Groq API key](https://console.groq.com/).

```bash
make install                       # npm install + uv sync

cp backend/.env.example backend/.env
# edit backend/.env → GROQ_API_KEY=...

make dev                           # backend :8000 + frontend :3000
```

Optional: set `GROQ_MODEL` in `backend/.env` to override the default
(`openai/gpt-oss-120b`), and `CHAT_API_URL` in `frontend/.env.local` if the
backend isn't on `http://localhost:8000`.

### Checks

```bash
make ci-cd          # everything the PR gates run
make test           # frontend Vitest + backend pytest
make lint           # eslint/prettier/tsc + ruff
```

Backend tests mock the RAG pipeline and the Groq client, so they run without
models, indexes, or API keys.

### Rebuilding the indexes

`backend/indexes/` (chunks + FAISS + BM25) is committed so deploys and cold
starts never re-embed the corpus. Rebuild only when `content/nietzsche/` or the
chunking parameters change:

```bash
make build-index    # chunks 1200/150, embeds, writes backend/indexes/
```

`backend/scripts/preprocess_texts.py` regenerates the cleaned texts from the
raw Project Gutenberg files.

## Deployment

- **Frontend — Vercel**: root directory `frontend`, framework preset Next.js.
  Env: `CHAT_API_URL` pointing at the backend.
- **Backend — Railway**: root directory `backend`, builder forced to
  **Dockerfile** (models are baked into the image so cold starts never hit
  Hugging Face). Env: `GROQ_API_KEY`, `ALLOWED_ORIGINS` (the Vercel origins),
  optionally `GROQ_MODEL`. Healthcheck path `/api/health` with a generous
  timeout (~300s) for first model load.

Both platforms deploy from `main` via their git integrations; CI on GitHub
Actions is advisory unless `main` is branch-protected.

## The corpus

19 public-domain works from [Project Gutenberg](https://www.gutenberg.org/),
including *Thus Spake Zarathustra*, *Beyond Good and Evil*, *The Genealogy of
Morals*, *The Antichrist*, *Ecce Homo*, *The Twilight of the Idols*, *The Birth
of Tragedy*, *The Joyful Wisdom*, *The Dawn of Day*, *Human, All-Too-Human*,
*The Will to Power*, and more. Per-work title, translator, and source URL live
in `content/nietzsche/metadata/sources.yaml` and are attached to every
retrieved passage.

## License

Open source. The Nietzsche texts are in the public domain.
