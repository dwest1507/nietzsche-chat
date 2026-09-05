# Agent guide

Two-service monorepo: `frontend/` (Next.js App Router, TypeScript, Tailwind v4)
and `backend/` (FastAPI, Python, uv). The corpus lives in `content/nietzsche/`
and the pre-built search indexes are committed in `backend/indexes/`.

## Conventions

- CI logic lives in `scripts/*.sh`; the `Makefile` and GitHub Actions workflows
  are thin callers. `make ci-cd` reproduces the PR gates locally.
- Frontend: ESLint + Prettier (no semicolons, single quotes), Vitest + RTL.
  Assistant chat messages render parsed markdown safely — never inject
  retrieved corpus text or raw HTML directly.
- Backend: Ruff (line-length 100), pytest with the RAG pipeline and Groq client
  mocked (`tests/conftest.py`) — tests must never load ML models or call APIs.
- The Nietzsche persona prompt in `backend/app/llm.py` is user-facing behavior;
  don't reword it casually.
- Rebuild indexes with `make build-index` only when `content/nietzsche/` or the
  chunking parameters change, and commit the artifacts.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`gh issue`); external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

