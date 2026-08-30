# Deployment

Two services, two platforms, one shared secret between them. Both deploy from `main`
through the platforms' own git integrations — GitHub Actions never deploys anything and
holds no deploy tokens.

Everything here is the state of the deployed system, not a plan: where a platform setting
is named, it is one you set in that platform's dashboard, because the repo carries no
`railway.json` and no `vercel.json`.

## Overview

| Piece                             | Platform                  | Deploys from                            | Root directory | Build                                                        |
| --------------------------------- | ------------------------- | --------------------------------------- | -------------- | ------------------------------------------------------------ |
| `frontend/` — Next.js App Router  | Vercel                    | push to `main`, Vercel git integration  | `frontend`     | Next.js framework preset                                     |
| `backend/` — FastAPI + RAG        | Railway (its own project) | push to `main`, Railway git integration | `backend`      | `backend/Dockerfile`, builder forced                         |
| `backend/indexes/` — FAISS + BM25 | none — committed to git   | —                                       | —              | built locally with `make build-index`, copied into the image |
| CI — lint, tests, build, audits   | GitHub Actions            | push to `main` and pull requests        | —              | never deploys; see [ci-cd.md](ci-cd.md)                      |

The browser only ever talks to the Next.js app. Its route handlers
(`frontend/app/api/chat/route.ts`, `frontend/app/api/ready/route.ts`) call the backend
through `frontend/lib/backendClient.ts`, which attaches the `X-Backend-Secret` header.
The backend's Railway URL is public, so that header is the only thing standing between it
and anyone who finds the URL — see
[ADR 0002](adr/0002-shared-secret-gateway.md).

## What a push to `main` does

```
                              push to main
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
  GitHub Actions              Railway                      Vercel
  (advisory unless        watch paths: backend/**    ignored build step:
   main is protected)            │                   skip unless frontend/
        │                        │                   changed
  lint · tests · build           │                          │
  audits · CodeQL          docker build from            next build,
        │                  backend/Dockerfile           deploy to production
        │                   (models baked in)                │
        │                        │                           │
   no deploy step         healthcheck GET /api/health   production alias
                          → deploy goes live            moves to the new build
                                 │
                          container may then
                          scale to zero on idle
```

CI is not a gate on either deploy. Railway and Vercel react to the push itself, so a red
build still ships unless `main` is branch-protected with the required checks listed in
[ci-cd.md](ci-cd.md).

## Environment variables

| Variable                | Set in                                                                              | What it is for                                                                                                                                                                          | Required                                                   |
| ----------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `GROQ_API_KEY`          | Railway, `backend/.env`                                                             | Groq API key, for condensing and generation                                                                                                                                             | **Yes**                                                    |
| `BACKEND_SHARED_SECRET` | Railway, Vercel (Production **and** Preview), `backend/.env`, `frontend/.env.local` | The secret the Next route sends as `X-Backend-Secret`; anything else gets a 401                                                                                                         | **Yes**, on both sides, identical                          |
| `CHAT_API_URL`          | Vercel (Production **and** Preview), `frontend/.env.local`                          | Base URL of the backend, e.g. `https://<service>.up.railway.app`. Server-side only                                                                                                      | **Yes in production**; defaults to `http://localhost:8000` |
| `GROQ_MODEL`            | Railway, `backend/.env`                                                             | Overrides the default model, `openai/gpt-oss-120b`                                                                                                                                      | No                                                         |
| `ALLOWED_ORIGINS`       | Railway, `backend/.env`                                                             | Comma-separated CORS origins; defence in depth only, see [below](#cors-is-not-what-guards-the-backend). Defaults to `http://localhost:3000`                                             | No                                                         |
| `SENTRY_DSN`            | Railway only                                                                        | Backend error reporting. Deliberately unset locally and in CI: with no DSN nothing is initialised and nothing is sent, so development failures never spend the free tier's event budget | No                                                         |
| `PORT`                  | Railway provides it                                                                 | The Dockerfile's `CMD` binds `${PORT:-8000}`                                                                                                                                            | Platform-provided                                          |

Two things to note about how these fail. `GROQ_API_KEY` and `BACKEND_SHARED_SECRET` are
read with `os.environ[...]`, so a missing one is loud — the container exits with a
`KeyError` before serving anything. `BACKEND_SHARED_SECRET` on the frontend is read with
`process.env.BACKEND_SHARED_SECRET ?? ''`, so a missing one is silent: the route sends an
empty header and every backend call comes back 401.

Neither frontend variable may ever be given a `NEXT_PUBLIC_` prefix. Both are server-side
only, and a prefix would ship the backend URL and the secret to the browser.

## First-time setup

The steps are ordered because each one produces a value the next one needs. Doing them
out of order means going back.

### 1. Verify the committed indexes — before touching either platform

`backend/indexes/` is committed, and the Dockerfile copies it into the image; nothing is
built on the platform. A deploy with missing or truncated index files builds and starts
happily, then fails warm-up and can never answer a question. Check first:

```bash
git ls-files backend/indexes
# backend/indexes/bm25.pkl
# backend/indexes/chunks.json
# backend/indexes/faiss.index

cd backend && uv run python -c "
import faiss, json, pickle
index = faiss.read_index('indexes/faiss.index')
chunks = json.load(open('indexes/chunks.json'))
bm25 = pickle.load(open('indexes/bm25.pkl', 'rb'))
print(index.ntotal, len(chunks), bm25.corpus_size)
"
# 7331 7331 7331
```

The three numbers must be equal — one vector and one BM25 document per chunk. If the
files are missing or the counts disagree, rebuild with `make build-index` and commit the
artifacts before deploying.

### 2. Deploy the backend to a new Railway project

Create a **new Railway project** on the existing account rather than adding a service to
the one that is already there. The two backends are unrelated, and separate projects keep
their variables, logs, usage and deletion independent.

Point it at this repository and set, on the service:

| Setting          | Value                              | Why                                                                                                                                                          |
| ---------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Root directory   | `backend`                          | Makes `backend/` the build context, which is what the Dockerfile's `COPY app ./app` and `COPY indexes ./indexes` expect                                      |
| Builder          | **Dockerfile** — set it explicitly | Railway's default builder wins otherwise, silently. See [the Railway trap](#the-railway-trap-the-dockerfile-that-was-never-used)                             |
| Watch paths      | `backend/**`                       | A frontend-only push must not rebuild a multi-gigabyte image                                                                                                 |
| Healthcheck path | `/api/health`                      | Liveness only: it answers as soon as the port binds, so a deploy never waits on the model load ([ADR 0003](adr/0003-readiness-is-separate-from-liveness.md)) |
| Scale to zero    | enabled                            | The service holds ~1–1.5 GB resident and Railway bills what is held ([ADR 0001](adr/0001-scale-to-zero-with-warm-on-arrival.md))                             |

Then set the variables — `GROQ_API_KEY`, `BACKEND_SHARED_SECRET`, and optionally
`GROQ_MODEL` and `SENTRY_DSN`. Generate the secret now; step 3 needs the same value:

```bash
openssl rand -hex 32
```

Leave `ALLOWED_ORIGINS` for step 4 — the Vercel origin does not exist yet. The backend
starts without it (it defaults to `http://localhost:3000`), and nothing in production
depends on it.

The first build takes a while: the image pre-downloads both sentence-transformers models
so container startup never reaches out to Hugging Face. Watch for that step in the build
log; its absence is the builder trap.

**Produces:** the service's public URL. Step 3 needs it.

### 3. Deploy the frontend to Vercel

Import the same repository as a Vercel project and set:

| Setting            | Value       | Why                                                                                                                                                 |
| ------------------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Root directory     | `frontend`  | The Next.js app is not at the repo root                                                                                                             |
| Framework preset   | **Next.js** | Anything else builds cleanly and serves a site where every route 404s. See [the Vercel trap](#the-vercel-trap-a-clean-build-where-every-route-404s) |
| Ignored build step | see below   | Stops backend-only pushes from rebuilding and redeploying the frontend                                                                              |

Environment variables, set for **Production and Preview both**:

- `CHAT_API_URL` — the Railway URL from step 2, no trailing slash.
- `BACKEND_SHARED_SECRET` — the exact value from step 2.

Preview deployments talk to the same production backend with the same secret. There is no
second backend to point them at, and a preview whose chat cannot answer is not much of a
preview. The cost of that choice is that preview traffic spends the same Groq allowance.

For the ignored build step, run this from the project's root directory (`frontend`):

```bash
git diff --quiet HEAD^ HEAD -- .
```

Vercel skips the build when the command exits 0 and builds when it exits non-zero.
`git diff --quiet` exits 0 when nothing under `frontend/` changed in the pushed commit and
1 when something did — so a backend-only push is skipped, and a commit where `HEAD^` is
unavailable errors out and builds, which is the safe direction to fail.

**Produces:** the production URL, and the preview URL pattern. Step 4 needs them.

### 4. Close the loop

Back in Railway, set `ALLOWED_ORIGINS` to the Vercel origins, comma-separated and with no
spaces:

```
https://your-app.vercel.app,https://your-app-git-main-you.vercel.app
```

This is defence in depth and nothing more (see below) — but a stale value is a
misleading thing to leave behind. Redeploy the backend so it picks the variable up.

Then smoke-test the deployed frontend, ideally on a backend that has been asleep:

1. Load the page. The status line reads _"Nietzsche is stirring…"_ while the container
   wakes and loads its models, then goes quiet.
2. Ask a question. Source passages appear first, then the answer streams in.
3. If you click a starter question during the wake, it is held and dispatched when the
   backend reports ready — that is the intended behaviour, not a hang.

### Not set up here: a custom domain

There is none. If one is added, it goes in Vercel (Project → Domains, plus the DNS
records Vercel prints), and then the new origin has to be added to `ALLOWED_ORIGINS` in
Railway. The backend keeps its Railway URL — it is never addressed by a browser, so it
gains nothing from a domain of its own.

## The two traps

Both of these produce a build log that looks entirely healthy. That is what makes them
expensive.

### The Railway trap: the Dockerfile that was never used

Railway's default builder (Railpack; Nixpacks on older projects) detects a Python project
and builds it its own way. It does this even though `backend/Dockerfile` is right there,
and the resulting build **succeeds** — it is simply a build of something else.

Recognise it by what is missing from the build log. A real Dockerfile build runs the steps
in `backend/Dockerfile`, and the loud one is the model pre-download —
`sentence_transformers` fetching `all-mpnet-base-v2` and `ms-marco-MiniLM-L-6-v2`, which
takes minutes and moves hundreds of megabytes. If the build finished quickly, mentions a
detected provider or install plan instead of the Dockerfile's steps, and never downloads a
model, the Dockerfile was ignored.

What follows at runtime: the models are not in the image, so the warm-up thread reaches
out to Hugging Face on every cold start (slow, and it depends on a third party being up),
and the container's start command is whatever the builder guessed rather than the
Dockerfile's `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.

**Fix:** service settings → Build → set the builder to **Dockerfile** explicitly, with
root directory `backend` so `Dockerfile` resolves. Redeploy and confirm the model download
appears in the log.

### The Vercel trap: a clean build where every route 404s

With the wrong framework preset, Vercel still runs the build — `next build` output and all
— and then serves the result as if it were a static directory. The build log is perfect
and the deployed site 404s on every route, including `/`.

The two Vercel error pages that look alike here mean different things:

| Error                       | Meaning                                                                         | Usual cause                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `404: NOT_FOUND`            | The deployment exists and is being served; it has no route matching the request | Wrong framework preset, or a root directory that does not contain the Next.js app                 |
| `404: DEPLOYMENT_NOT_FOUND` | There is no deployment behind this URL at all                                   | A stale or mistyped preview URL, a deleted deployment, or a domain not assigned to any deployment |

`NOT_FOUND` is a build-configuration problem; `DEPLOYMENT_NOT_FOUND` means you are looking
at the wrong URL. Fixing the preset will not help the second one.

**Fix for `NOT_FOUND`:** project settings → Build & Deployment → framework preset
**Next.js**, root directory `frontend`. Then redeploy — changing the setting does not
rebuild the existing deployment.

## Things that are easy to get wrong later

### The shared secret lives in two places

`BACKEND_SHARED_SECRET` is set in Railway and in Vercel, and rotating it means rotating it
in both. If the two drift, **every chat request fails**, and the failure does not look
like an authentication problem from the outside:

- The frontend polls `/api/ready`, the backend answers 401, and the route reports the
  pipeline as `failed` — a terminal state, so polling stops.
- Every question is then held and immediately turned into an error the visitor reads as
  _"Nietzsche could not be roused — his library failed to load. Please try again in a few
  minutes."_ The library is fine. The secret is wrong.
- A tab that was already open with a ready backend when the drift began sees the other
  copy instead: the chat POST gets a 401, the route maps it to a 502, and the visitor
  reads _"Nietzsche is unreachable at the moment."_
- Nothing in either message says "unauthorized", and no request reaches Groq, so a quiet
  Sentry project is not evidence that things are fine.

To rotate: set the new value in Railway, set the same value in Vercel for **both**
Production and Preview, then redeploy both. There is a window between the two where the
site is down; keep it short and do it deliberately.

### CORS is not what guards the backend

No browser request ever reaches the backend. Every call comes from the Vercel route
handler, server-side, where CORS does not apply. `ALLOWED_ORIGINS` therefore guards a path
nothing uses. It stays configured as defence in depth, but the shared secret is what
protects the service — and nothing should be built on the assumption that `ALLOWED_ORIGINS`
does more than it does. Widening it to `*` would not open a hole; narrowing it would not
close one. See [ADR 0002](adr/0002-shared-secret-gateway.md).

### Keep-warm pings defeat scale-to-zero

Pinging `/api/health` on a short interval — UptimeRobot every five minutes, a cron job,
anything of that shape — is not a middle ground between cold starts and always-on. It
prevents the container from ever sleeping, so it costs the same as always-on while adding
a moving part that can fail on its own. The cold start is meant to be hidden, not
prevented: the frontend pings readiness when the chat mounts, and the container wakes while
the visitor is still reading the page. See [ADR 0001](adr/0001-scale-to-zero-with-warm-on-arrival.md).

If traffic ever becomes steady enough that the container rarely sleeps, the honest move is
always-on, not a pinger.

### The rate limiter forgets when the backend sleeps

The per-visitor limit (`10/minute;100/day`, keyed on the address the frontend forwards as
`X-Client-IP`) is held in the process's own memory. It is lost on every restart, and the
backend restarts every time it wakes from sleep. Someone who spends their daily allowance,
waits for the container to sleep, and comes back gets a fresh one.

That is accepted, not a defect. The limit exists to keep one visitor from draining the
shared Groq quota in an afternoon, and it still does that; the alternative is an external
store — another service, another variable, another thing to pay for — for a portfolio
piece. Worth knowing before someone reads the counters as authoritative.

## When it goes wrong

### Railway

| Symptom                                                                                                                | Cause                                                                                                        | Fix                                                                      |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Container exits at startup, log ends in `KeyError: 'GROQ_API_KEY'`                                                     | The variable is not set on the service                                                                       | Set it, redeploy                                                         |
| Same, with `KeyError: 'BACKEND_SHARED_SECRET'`                                                                         | Same                                                                                                         | Set it, redeploy — and set the identical value in Vercel                 |
| Build succeeds in a couple of minutes and never downloads a model                                                      | The Dockerfile was ignored                                                                                   | [The Railway trap](#the-railway-trap-the-dockerfile-that-was-never-used) |
| Deploy never goes healthy                                                                                              | Healthcheck path is wrong — the routers are mounted under `/api`                                             | Set the healthcheck path to `/api/health`                                |
| Log shows `RAG pipeline warm-up failed; the first chat request will retry` with a `FileNotFoundError` under `indexes/` | The index files are not in the image                                                                         | Verify them (step 1), commit, redeploy                                   |
| Log shows `RAG pipeline warm-up failed` with a Hugging Face or network error                                           | Models are not baked in — usually the builder trap, occasionally a Hub outage on a first run                 | Force the Dockerfile builder and redeploy                                |
| Log shows `SENTRY_DSN is unset; backend error reporting is disabled`                                                   | Exactly what it says; expected everywhere but production                                                     | Set `SENTRY_DSN` in Railway if you want production errors reported       |
| Log shows `RAG pipeline ready`                                                                                         | Healthy — the backend can answer                                                                             | —                                                                        |
| Usage climbs with no visitors                                                                                          | Something is pinging the service and holding it awake; `/api/health` is public and unauthenticated by design | Remove the pinger; see the keep-warm note above                          |

### Vercel

| Symptom                                               | Cause                                                          | Fix                                                                            |
| ----------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Every route returns `404: NOT_FOUND`, build log clean | Wrong framework preset or root directory                       | [The Vercel trap](#the-vercel-trap-a-clean-build-where-every-route-404s)       |
| `404: DEPLOYMENT_NOT_FOUND`                           | The URL has no deployment behind it                            | Open the deployment from the Vercel dashboard; check domain assignment         |
| Frontend rebuilds on a backend-only push              | Ignored build step missing or wrong                            | Set it as in step 3                                                            |
| Production works, previews fail on every question     | Variables set for Production only                              | Add `CHAT_API_URL` and `BACKEND_SHARED_SECRET` to the Preview environment too  |
| A new variable has no effect                          | Variables are read at build and run time of a given deployment | Redeploy; changing a variable does not retroactively change a built deployment |

### What the visitor sees, and what it means

The copy is in the app's voice, so the mapping is worth writing down.

| The visitor reads                                               | Actually means                                                                                                                      | Where to look                                                                              |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| _"Nietzsche is stirring…"_ and it never resolves                | `/api/ready` cannot reach the backend at all, so the frontend reads it as still waking                                              | `CHAT_API_URL` in Vercel; is the Railway service deployed and healthy?                     |
| _"Nietzsche could not be roused — his library failed to load."_ | The backend answered the readiness probe with something that is not OK — a 401 from a drifted secret, or a genuinely failed warm-up | Railway log: `RAG pipeline ready` means the pipeline is fine and the secret is the problem |
| _"Nietzsche is unreachable at the moment."_                     | The chat request failed: the backend was unreachable, or answered with a status the route maps to 502                               | Railway log for a traceback; Sentry if it is configured                                    |
| _"Even Zarathustra needed rest — too many questions at once."_  | That visitor hit their own rate limit (10/minute or 100/day), rejected with HTTP 429 before any stream started                      | Nothing to fix; it is the limiter working                                                  |
| _"Nietzsche has spoken his fill for today."_                    | The service-wide Groq allowance is spent — the backend sent the `provider_quota` error category                                     | Groq console; it renews on Groq's schedule                                                 |

## Everyday deploys

Merge to `main`. Then:

- A change under `frontend/` builds and deploys on Vercel; the backend is untouched.
- A change under `backend/` rebuilds the Railway image; the frontend is untouched.
- A change to both deploys both, independently and not atomically. If the two must land
  together — a change to the stream protocol, say — ship the backwards-compatible half
  first.
- A change to `content/nietzsche/` does nothing on its own. Rebuild the indexes with
  `make build-index` and commit the artifacts; that lands under `backend/` and triggers a
  Railway build.
- A change only to `scripts/`, `Makefile`, `docs/` or `README.md` deploys nothing.

Watch a backend deploy for `RAG pipeline ready` in the Railway log before calling it done.
The deploy goes green as soon as the port binds — `/api/health` says nothing about the
models, deliberately.

## Rollback

**Vercel.** Deployments → the last known-good production deployment → promote it to
production. The alias moves immediately; there is no rebuild. This is the fast one, and it
is the right first move whenever the frontend is the suspect.

**Railway.** Deployments → the last known-good deployment → redeploy/roll back to it. It
restores that image, which takes as long as a container start plus a model load. Two
things it does _not_ restore: environment variables, which are service state rather than
part of the deployment, and anything on the other platform.

**Both.** `git revert` on `main` and push — this rolls back through the normal path and
leaves the repo agreeing with what is deployed, which the dashboard rollbacks do not.
Slower, and worth it for anything you are not going to fix within the hour.

One caution: if the change you are rolling back touched `BACKEND_SHARED_SECRET`, rolling
back one platform re-creates the drift. Roll back the secret in both, or neither.

## Local environment

```bash
make install                       # uv sync + npm install

cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
# backend/.env    → GROQ_API_KEY=<your key>
# both files      → the same BACKEND_SHARED_SECRET (any non-empty value)

make dev                           # backend :8000, frontend :3000
```

Notes on how local differs from deployed:

- The backend refuses to start without `GROQ_API_KEY` or `BACKEND_SHARED_SECRET`; the two
  files must agree or every request 401s, exactly as in production.
- `CHAT_API_URL` defaults to `http://localhost:8000`, so it only needs setting if the
  backend is somewhere else.
- `SENTRY_DSN` stays unset. `scripts/backend-test.sh` goes further and exports it as empty,
  so a developer's `backend/.env` cannot make the test suite report its own deliberate
  failures to the real project.
- There is no cold start and no scale-to-zero locally: the models load once, in a
  background thread, on startup. The first `/api/ready` may report `loading` for tens of
  seconds — that is the same code path production uses.
- `make build-index` is the only command that writes to `backend/indexes/`, and its output
  is committed.

Run `make ci-cd` before pushing; it runs the same scripts CI does.

## Checklist

First deploy:

- [ ] `git ls-files backend/indexes` lists all three files, and the FAISS, chunk and BM25
      counts agree
- [ ] New Railway project, root directory `backend`, builder forced to **Dockerfile**
- [ ] Railway watch paths `backend/**`, healthcheck `/api/health`, scale-to-zero on
- [ ] Railway variables: `GROQ_API_KEY`, `BACKEND_SHARED_SECRET`, optionally `GROQ_MODEL`
      and `SENTRY_DSN`
- [ ] Build log shows the sentence-transformers model download (proof the Dockerfile ran)
- [ ] Railway log shows `RAG pipeline ready`
- [ ] Vercel project, root directory `frontend`, framework preset **Next.js**
- [ ] Vercel variables `CHAT_API_URL` and `BACKEND_SHARED_SECRET`, set for Production
      **and** Preview, with the secret identical to Railway's
- [ ] Vercel ignored build step configured
- [ ] `ALLOWED_ORIGINS` in Railway updated with the Vercel origins, backend redeployed
- [ ] A backend-only push does not rebuild the frontend
- [ ] Cold-start check: sleep the backend, load the page, watch it wake and answer
- [ ] `main` branch-protected with the checks in [ci-cd.md](ci-cd.md)

After any change to the shared secret:

- [ ] New value in Railway
- [ ] Same value in Vercel, Production and Preview
- [ ] Both redeployed
- [ ] A question answered end to end on the deployed site
