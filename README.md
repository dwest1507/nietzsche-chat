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
request and proxies it to the FastAPI backend through `frontend/lib/backendClient.ts`
(`CHAT_API_URL` and the `BACKEND_SHARED_SECRET` header, both server-side only),
piping the token stream back unchanged. The backend's public URL rejects any
request that does not present that shared secret — see
`docs/adr/0002-shared-secret-gateway.md`.

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
3:{"category": "provider_quota"|"generic"}         error (replaces d:)
```

The `3:` line carries a failure *category*, never the upstream error text —
provider messages and tracebacks stay in the server log:

| category | meaning |
| --- | --- |
| `provider_quota` | the service-wide Groq allowance is spent; try again later |
| `generic` | anything else went wrong |

Clients must treat an unrecognised category as `generic`, so a category added
later degrades instead of breaking the stream.

## 🏷️ Versioning

This project uses [Release Please](https://github.com/googleapis/release-please) for
automated versioning and changelog generation.

1. **Develop:** merge changes into `main` using
   [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, ...).
2. **Release PR:** the `Release Please` workflow opens a PR bumping `version.txt` and
   updating `CHANGELOG.md`.
3. **Merge:** merging that PR creates the git tag and the GitHub Release.

The current version lives in `version.txt`; the release history is in `CHANGELOG.md`.

## 📄 License

A visitor's own rate limit is not one of these: it is rejected with **HTTP 429**
before the stream starts, so it never reaches the response body.

## Development

Prerequisites: [uv](https://docs.astral.sh/uv/), Node.js (see
`frontend/.nvmrc`), and a free [Groq API key](https://console.groq.com/).

```bash
make install                       # npm install + uv sync

cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
# edit backend/.env → GROQ_API_KEY=...
# set the same BACKEND_SHARED_SECRET in both files (any non-empty value locally)

make dev                           # backend :8000 + frontend :3000
```

`BACKEND_SHARED_SECRET` is required on both sides: the backend refuses to start
without it, and if the two values drift every chat request fails with 401.

Optional: set `GROQ_MODEL` in `backend/.env` to override the default
(`openai/gpt-oss-120b`), and `CHAT_API_URL` in `frontend/.env.local` if the
backend isn't on `http://localhost:8000`.

### Checks

```bash
make ci-cd          # everything the PR gates run
make test           # frontend Vitest + backend pytest
make lint           # eslint/prettier/tsc + ruff
make lighthouse     # Lighthouse CI budget check (needs Chrome)
```

Backend tests mock the RAG pipeline and the Groq client, so they run without
models, indexes, or API keys. `make ci-cd` also runs a Lighthouse CI pass
against the production build (`frontend/lighthouserc.json`), asserting
≥ 0.9 on accessibility / best-practices / SEO (performance warns).

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

Frontend on Vercel, backend on Railway, both deploying from `main` through the
platforms' own git integrations. First-time setup, every environment variable,
the platform traps, failure symptoms and rollback live in one place:
**[docs/deployment.md](docs/deployment.md)**.

CI is described in [docs/ci-cd.md](docs/ci-cd.md) — it never deploys, and holds
no deploy tokens.

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
