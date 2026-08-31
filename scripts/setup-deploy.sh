#!/usr/bin/env bash
# Setup wizard for the deployment steps that only happen in someone's browser:
# the Railway project, the Vercel project, the Sentry project, and branch
# protection. It walks them one at a time, says what to set and what value to
# enter, waits for you, and verifies what can be verified from here.
#
# Called by `make setup-deploy`; nothing in CI runs it.
#
# This is the doing counterpart to docs/deployment.md — that document holds the
# reasons, the two traps and the failure tables, and this script points at them
# rather than repeating them. It writes nothing to either platform and nothing
# to a tracked file, so re-running it against a live deployment is a safe way to
# check that deployment is still correct. `--verify-only` is that check alone.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

verify_only=0
backend_url=""
frontend_url=""
shared_secret="${BACKEND_SHARED_SECRET:-}"
ready_wait=120   # seconds readiness may stay "loading" — a cold start loads models
http_timeout=60  # per-request curl timeout
chat_timeout=180 # a chat request may have to wake the container first

usage() {
  cat <<'USAGE'
usage: setup-deploy.sh [--verify-only] [options]

Walks the manual dashboard steps of a first deployment — Railway, Vercel,
Sentry, branch protection — pausing at each one, and verifies the result.
Nothing is written to either platform and nothing is written to a tracked
file, so running it again on a working deployment is safe.

Options:
  --verify-only        Run the checks against an existing deployment and exit.
                       No prompts, nothing generated, nothing written.
  --backend-url URL    Railway backend base URL, e.g. https://x.up.railway.app
  --frontend-url URL   Vercel frontend base URL, e.g. https://x.vercel.app
  --secret VALUE       The shared secret, for the readiness check. Prefer the
                       BACKEND_SHARED_SECRET environment variable — an argument
                       is visible to anyone who can list processes.
  --wait SECONDS       How long readiness may stay "loading" (default 120).
  -h, --help           This text.

Checks, all read-only:
  GET  <backend>/api/health   liveness: 200 {"status":"ok"}
  GET  <backend>/api/ready    readiness, with the X-Backend-Secret header —
                              a 401 here means the two secrets do not match
  POST <frontend>/api/chat    end to end through the deployed frontend

Exit status: 0 every check passed (warnings allowed), 1 a check failed,
2 bad usage or a missing prerequisite.

Why any of it is set up this way, and what to do when it breaks:
docs/deployment.md. The status-check names: docs/ci-cd.md.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --verify-only) verify_only=1 ;;
    --backend-url) backend_url="${2:-}"; shift ;;
    --frontend-url) frontend_url="${2:-}"; shift ;;
    --secret) shared_secret="${2:-}"; shift ;;
    --wait) ready_wait="${2:-}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown option '$1'" >&2
      echo "try: $0 --help" >&2
      exit 2
      ;;
  esac
  shift
done

case "$ready_wait" in
  ''|*[!0-9]*) echo "error: --wait takes a whole number of seconds" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Output and prompt helpers
# ---------------------------------------------------------------------------

say() { printf '%s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
blank() { printf '\n'; }

step() {
  printf '\n==> %s\n' "$*"
}

pass_count=0
warn_count=0
fail_count=0

result_pass() { pass_count=$((pass_count + 1)); printf '    [ ok ] %s\n' "$*"; }
result_warn() { warn_count=$((warn_count + 1)); printf '    [warn] %s\n' "$*"; }
result_fail() { fail_count=$((fail_count + 1)); printf '    [FAIL] %s\n' "$*"; }

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

confirm() {
  local reply=""
  read -r -p "    $1 [y/N] " reply || reply=""
  [[ "$(trim "$reply")" =~ ^[Yy] ]]
}

pause() {
  local reply=""
  read -r -p "    ${1:-Press Enter when that is done} " reply || true
}

ask() { # ask VARNAME "prompt"
  local __var="$1" reply=""
  read -r -p "    $2: " reply || reply=""
  printf -v "$__var" '%s' "$(trim "$reply")"
}

ask_url() { # ask_url VARNAME "prompt"
  local __var="$1" prompt="$2" reply=""
  while :; do
    read -r -p "    $prompt: " reply || reply=""
    reply="$(trim "$reply")"
    while [ -n "$reply" ] && [ "${reply%/}" != "$reply" ]; do reply="${reply%/}"; done
    if [ -z "$reply" ]; then
      note "That value is needed to go on."
      continue
    fi
    case "$reply" in
      https://*) ;;
      http://*) note "Note: that is http:// — both platforms serve https://." ;;
      *) note "Include the scheme, e.g. https://something.up.railway.app"; continue ;;
    esac
    printf -v "$__var" '%s' "$reply"
    return 0
  done
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

if [ ! -f "$repo_root/backend/Dockerfile" ]; then
  echo "error: run this from the nietzsche-chat repository (backend/Dockerfile not found)" >&2
  exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required — every check here is an HTTP request" >&2
  exit 2
fi

if [ "$verify_only" -eq 0 ] && [ ! -t 0 ]; then
  echo "error: the wizard is interactive; run it from a terminal, or use --verify-only" >&2
  exit 2
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/nietzsche-setup-XXXXXX")"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

# Normalise anything that arrived on the command line the same way ask_url does.
for __name in backend_url frontend_url; do
  __value="$(trim "${!__name}")"
  while [ -n "$__value" ] && [ "${__value%/}" != "$__value" ]; do __value="${__value%/}"; done
  printf -v "$__name" '%s' "$__value"
done
unset __name __value

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

HTTP_STATUS=""
HTTP_BODY=""
HTTP_ERROR=""

http_request() { # http_request METHOD URL TIMEOUT [extra curl args...]
  local method="$1" url="$2" timeout="$3"
  shift 3
  local body_file="$tmpdir/body" err_file="$tmpdir/curl.err"
  : >"$body_file"
  : >"$err_file"
  HTTP_STATUS="$(curl -sS -o "$body_file" -w '%{http_code}' \
    --max-time "$timeout" -X "$method" "$@" "$url" 2>"$err_file")" || HTTP_STATUS="000"
  HTTP_BODY="$(cat "$body_file")"
  HTTP_ERROR="$(cat "$err_file")"
}

unreachable_note() {
  note "${HTTP_ERROR:-curl gave no response at all}"
  note "A sleeping Railway container still answers — the platform wakes it — so this"
  note "usually means the URL is wrong or nothing is deployed behind it."
}

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

check_health() { # check_health BACKEND_URL
  local url="$1" attempt
  note "GET $url/api/health"
  for attempt in 1 2 3; do
    http_request GET "$url/api/health" "$http_timeout"
    if [ "$HTTP_STATUS" = "200" ] && printf '%s' "$HTTP_BODY" | tr -d ' \n' | grep -q '"status":"ok"'; then
      result_pass "backend liveness: 200 {\"status\":\"ok\"}"
      return 0
    fi
    if [ "$attempt" -lt 3 ]; then
      note "attempt $attempt: ${HTTP_STATUS} — retrying (the container may be waking)"
      sleep 5
    fi
  done

  result_fail "backend liveness: $url/api/health did not answer 200 {\"status\":\"ok\"}"
  case "$HTTP_STATUS" in
    000) unreachable_note ;;
    404)
      note "404: the routers are mounted under /api. Check the base URL has no path of"
      note "its own, and that Railway's healthcheck path is /api/health — not /health."
      ;;
    502|503)
      note "The service is reachable but not serving. Railway log: a KeyError for"
      note "GROQ_API_KEY or BACKEND_SHARED_SECRET means the variable is not set."
      ;;
    *) note "status $HTTP_STATUS, body: ${HTTP_BODY:0:200}" ;;
  esac
  note "See docs/deployment.md → When it goes wrong → Railway."
  return 1
}

check_ready() { # check_ready BACKEND_URL SECRET
  local url="$1" secret="$2" deadline state
  if [ -z "$secret" ]; then
    result_warn "backend readiness: skipped, no shared secret given (--secret or BACKEND_SHARED_SECRET)"
    return 0
  fi

  note "GET $url/api/ready  (header X-Backend-Secret)"
  deadline=$(( $(date +%s) + ready_wait ))
  while :; do
    http_request GET "$url/api/ready" "$http_timeout" -H "X-Backend-Secret: $secret"
    case "$HTTP_STATUS" in
      200)
        state="$(printf '%s' "$HTTP_BODY" | tr -d ' \n' | sed -n 's/.*"status":"\([a-z]*\)".*/\1/p')"
        case "$state" in
          ready)
            result_pass "backend readiness: ready — the secret matches and the pipeline is loaded"
            return 0
            ;;
          loading)
            if [ "$(date +%s)" -ge "$deadline" ]; then
              result_warn "backend readiness: still \"loading\" after ${ready_wait}s"
              note "A cold start loads two models; that is slow but finite. If it never"
              note "resolves, the Railway log will say why — look for 'RAG pipeline ready'."
              return 0
            fi
            note "loading… (waking and loading models; waiting up to ${ready_wait}s)"
            sleep 10
            ;;
          failed)
            result_fail "backend readiness: \"failed\" — the warm-up raised"
            note "Railway log: 'RAG pipeline warm-up failed'. A FileNotFoundError under"
            note "indexes/ means the index files are not in the image; a Hugging Face or"
            note "network error usually means the Dockerfile was never used."
            note "See docs/deployment.md → The Railway trap."
            return 1
            ;;
          *)
            result_fail "backend readiness: unrecognised body: ${HTTP_BODY:0:200}"
            return 1
            ;;
        esac
        ;;
      401)
        result_fail "backend readiness: 401 — the shared secret does not match"
        note "BACKEND_SHARED_SECRET in Railway and in Vercel (Production and Preview)"
        note "must be the same string. This 401 is the earliest honest signal of drift:"
        note "in the browser it shows up as 'Nietzsche could not be roused'."
        note "See docs/deployment.md → The shared secret lives in two places."
        return 1
        ;;
      000)
        result_fail "backend readiness: no response from $url/api/ready"
        unreachable_note
        return 1
        ;;
      *)
        result_fail "backend readiness: status $HTTP_STATUS, body: ${HTTP_BODY:0:200}"
        return 1
        ;;
    esac
  done
}

check_chat() { # check_chat FRONTEND_URL
  local url="$1" payload attempt error_line
  payload='{"message":"In one sentence, what is amor fati?","history":[]}'

  note "POST $url/api/chat  (a real question, through the deployed frontend)"
  for attempt in 1 2; do
    http_request POST "$url/api/chat" "$chat_timeout" \
      -H 'Content-Type: application/json' --data "$payload"
    if [ "$HTTP_STATUS" = "200" ] || [ "$attempt" -eq 2 ]; then
      break
    fi
    note "attempt $attempt: ${HTTP_STATUS} — retrying once; the backend may have been asleep"
    sleep 15
  done

  if [ "$HTTP_STATUS" = "200" ]; then
    error_line="$(printf '%s\n' "$HTTP_BODY" | grep -m1 '^3:' || true)"
    if printf '%s\n' "$HTTP_BODY" | grep -q '^0:"'; then
      result_pass "end to end: sources and a streamed answer came back through the frontend"
      printf '%s\n' "$HTTP_BODY" | grep -q '^2:\[' \
        || result_warn "end to end: no 2: sources line — the answer was not grounded in the corpus"
      return 0
    fi
    case "$error_line" in
      *provider_quota*)
        result_warn "end to end: the stream ended with provider_quota"
        note "The wiring is right — the request reached Groq — but the service-wide"
        note "allowance is spent. Visitors read 'Nietzsche has spoken his fill for"
        note "today.' Check the Groq console; it renews on Groq's schedule."
        return 0
        ;;
      *generic*)
        result_fail "end to end: the stream ended with the generic error category"
        note "The backend raised and kept the detail to itself, by design. The"
        note "traceback is in the Railway log, and in Sentry if SENTRY_DSN is set."
        return 1
        ;;
      *)
        result_fail "end to end: 200, but no tokens arrived"
        note "body (first 200 chars): ${HTTP_BODY:0:200}"
        return 1
        ;;
    esac
  fi

  result_fail "end to end: POST $url/api/chat returned $HTTP_STATUS"
  case "$HTTP_STATUS" in
    404)
      note "404 from the frontend's own route is the Vercel trap: with the wrong"
      note "framework preset or root directory the build is clean and every route"
      note "404s. See docs/deployment.md → The Vercel trap."
      ;;
    429)
      note "That is the per-visitor rate limit (10/minute, 100/day) — the limiter"
      note "working, not a fault. Wait a minute and run --verify-only again."
      ;;
    502)
      note "The frontend could not get a usable answer from the backend. Either"
      note "CHAT_API_URL is wrong or unset in Vercel, or the secrets have drifted"
      note "(the backend answers 401 and this route maps it to 502)."
      ;;
    000) unreachable_note ;;
    *) note "body (first 200 chars): ${HTTP_BODY:0:200}" ;;
  esac
  note "See docs/deployment.md → When it goes wrong."
  return 1
}

run_checks() {
  local ran=0
  if [ -n "$backend_url" ]; then
    step "Checking the backend"
    check_health "$backend_url" || true
    check_ready "$backend_url" "$shared_secret" || true
    ran=1
  fi
  if [ -n "$frontend_url" ]; then
    step "Checking the frontend, end to end"
    check_chat "$frontend_url" || true
    ran=1
  fi
  if [ "$ran" -eq 0 ]; then
    say "Nothing to check: give --backend-url and/or --frontend-url."
    return 2
  fi
  return 0
}

summarise() {
  blank
  say "-----------------------------------------------------------------"
  say "Checks: $pass_count passed, $warn_count warned, $fail_count failed"
  if [ "$fail_count" -gt 0 ]; then
    say "Something is not right — the notes above say where to look, and"
    say "docs/deployment.md says why it fails that way."
    return 1
  fi
  say "Nothing failed."
  return 0
}

# ---------------------------------------------------------------------------
# --verify-only: the checks on their own, no prompts
# ---------------------------------------------------------------------------

if [ "$verify_only" -eq 1 ]; then
  say "nietzsche-chat — verifying a deployment (read-only, changes nothing)"
  run_checks || exit 2
  summarise || exit 1
  exit 0
fi

# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------

generate_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import secrets; print(secrets.token_hex(32))'
  elif [ -r /dev/urandom ] && command -v od >/dev/null 2>&1; then
    # head -c takes a fixed count, so nothing here is killed by a broken pipe.
    head -c 32 /dev/urandom | od -An -v -tx1 | tr -d ' \n'
    printf '\n'
  else
    return 1
  fi
}

sentry_state="skipped"
protection_state="skipped"

cat <<'INTRO'
nietzsche-chat — first-time deployment wizard

Two services on two platforms, one shared secret between them. This walks the
parts an agent cannot do for you: the Railway project, the Vercel project, the
Sentry project, and branch protection. At each step it says what to set, waits
for you, and checks the result where a check is possible.

Three things worth knowing before you start:

  * It changes nothing. It prompts, checks and reports — so it is also safe to
    run again later to confirm a working deployment is still correct
    (`make setup-deploy` again, or scripts/setup-deploy.sh --verify-only).
  * The two platforms each need a value the other produces: the backend needs
    the frontend's origin, the frontend needs the backend's URL. So the last
    step goes back to Railway. That is the shape of the thing, not a mistake.
  * Dashboard wording moves. Where a label is quoted below it is described as
    well; go by the description if the platform has renamed something.

The reasons behind every setting, both build traps, and what each failure looks
like to a visitor: docs/deployment.md.
INTRO

pause "Press Enter to begin"

# --- 1. indexes -------------------------------------------------------------

step "Step 1/8  The committed indexes — before touching either platform"
note "backend/indexes/ is committed and the Dockerfile copies it into the image."
note "Nothing is built on the platform, and a deploy missing them starts happily"
note "and can then never answer a question."
blank
tracked="$(cd "$repo_root" && git ls-files backend/indexes)"
for f in backend/indexes/bm25.pkl backend/indexes/chunks.json backend/indexes/faiss.index; do
  if printf '%s\n' "$tracked" | grep -qx "$f"; then
    result_pass "tracked: $f"
  else
    result_fail "not tracked by git: $f — rebuild with 'make build-index' and commit"
  fi
done
blank
note "That they exist is not that they agree. docs/deployment.md step 1 has the"
note "one-liner that prints the FAISS, chunk and BM25 counts; all three must match."
if confirm "Run that count check now? It needs uv and takes a moment."; then
  if (cd "$repo_root/backend" && uv run python -c "
import faiss, json, pickle
index = faiss.read_index('indexes/faiss.index')
chunks = json.load(open('indexes/chunks.json'))
bm25 = pickle.load(open('indexes/bm25.pkl', 'rb'))
print('faiss', index.ntotal, 'chunks', len(chunks), 'bm25', bm25.corpus_size)
assert index.ntotal == len(chunks) == bm25.corpus_size, 'counts disagree'
"); then
    result_pass "index counts agree"
  else
    result_fail "index check failed — rebuild with 'make build-index' and commit the artifacts"
  fi
fi

# --- 2. the shared secret ---------------------------------------------------

step "Step 2/8  The shared secret"
note "BACKEND_SHARED_SECRET is the only thing between the public Railway URL and"
note "anyone who finds it. The Next.js route sends it as the X-Backend-Secret"
note "header; the backend 401s anything else. It has to be byte-for-byte the same"
note "in Railway and in Vercel — see docs/adr/0002-shared-secret-gateway.md."
blank
if [ -n "$shared_secret" ]; then
  note "Using the secret already given to this run; nothing new generated."
elif confirm "Is this deployment already configured with a secret you want to keep?"; then
  while [ -z "$shared_secret" ]; do
    read -r -s -p "    Paste it (not echoed, used only for the readiness check): " shared_secret || shared_secret=""
    printf '\n'
    shared_secret="$(trim "$shared_secret")"
  done
  note "Kept in memory for this run only. It is never written anywhere by this script."
else
  if ! shared_secret="$(generate_secret)"; then
    say "error: no way to generate a secret here (no openssl, no python3, no /dev/urandom)." >&2
    say "Generate 32 random bytes as hex elsewhere and re-run with --secret." >&2
    exit 2
  fi
  blank
  note "Your shared secret — copy it now, it is shown once and stored nowhere:"
  blank
  printf '        %s\n' "$shared_secret"
  blank
  note "It goes into Railway (BACKEND_SHARED_SECRET) in step 4 and into Vercel"
  note "(BACKEND_SHARED_SECRET, Production AND Preview) in step 5. Identical in"
  note "both. If the two ever drift, every chat request fails and none of the"
  note "error messages says so — docs/deployment.md spells that failure out."
fi
blank
pause "Press Enter when you have it somewhere safe"

# --- 3. Sentry --------------------------------------------------------------

step "Step 3/8  Sentry — optional, but do it before Railway"
note "SENTRY_DSN is set in Railway and nowhere else. It is deliberately unset"
note "locally and in CI: with no DSN nothing is initialised and nothing is sent,"
note "so development failures never spend the free tier's event budget."
note "Unset in production is legal too — the backend logs 'SENTRY_DSN is unset;"
note "backend error reporting is disabled' and carries on."
blank
if confirm "Set up Sentry error reporting?"; then
  note "In Sentry:"
  note "  1. Create a project. Platform: Python (FastAPI if it is offered — the"
  note "     choice only shapes the onboarding snippet, which you do not need:"
  note "     the SDK is already wired up in backend/app/errors.py)."
  note "  2. Open the project's settings and find its client key. The value you"
  note "     want is the DSN — an https:// URL with a key in it."
  note "  3. Copy it. In step 4 it becomes SENTRY_DSN in Railway."
  blank
  note "Do not paste the DSN into this terminal, into backend/.env, or into any"
  note "file in the repo. Railway is the only place it belongs."
  pause "Press Enter when you have the DSN"
  sentry_state="configured"
else
  note "Skipping. Backend errors will only reach the Railway log — which belongs"
  note "to a container that may have scaled to zero by the time you look."
  sentry_state="skipped"
fi

# --- 4. Railway -------------------------------------------------------------

step "Step 4/8  Railway — the backend"
note "Create a NEW Railway project rather than adding a service to an existing"
note "one, and point it at this repository. Separate projects keep variables,"
note "logs, usage and deletion independent."
blank
note "On the service, set:"
note "  Root directory    backend"
note "                    (the Dockerfile's COPY app / COPY indexes expect it as"
note "                     the build context)"
note "  Builder           Dockerfile — set it explicitly, in the build settings."
note "                    Railway's own builder wins otherwise, silently, and the"
note "                    build still succeeds. It is just a build of something"
note "                    else. See docs/deployment.md → The Railway trap."
note "  Watch paths       backend/**"
note "                    (a frontend-only push must not rebuild a multi-GB image)"
note "  Healthcheck path  /api/health"
note "                    (liveness only — it answers as soon as the port binds,"
note "                     so a deploy never waits on the model load)"
note "  Scale to zero     enabled"
note "                    (the service holds ~1-1.5 GB resident and Railway bills"
note "                     what is held)"
blank
note "Variables on the service:"
note "  GROQ_API_KEY            required — the container exits with a KeyError without it"
note "  BACKEND_SHARED_SECRET   required — the value from step 2, exactly"
note "  GROQ_MODEL              optional — defaults to openai/gpt-oss-120b"
note "  SENTRY_DSN              optional — the DSN from step 3"
note "  ALLOWED_ORIGINS         leave it for now; step 6 sets it. The Vercel"
note "                          origin does not exist yet, and the default"
note "                          (http://localhost:3000) breaks nothing."
note "  PORT                    Railway provides it; do not set it yourself."
blank
note "The first build is slow on purpose: the image pre-downloads both"
note "sentence-transformers models so a cold start never reaches Hugging Face."
note "Watch for that download in the build log — its absence IS the builder trap."
note "Then watch the deploy log for 'RAG pipeline ready'."
blank
pause "Press Enter when the deploy is live"

blank
note "Railway calls this the service's public domain — generate one if the"
note "service has none. It is the value Vercel needs as CHAT_API_URL."
ask_url backend_url "Backend URL (no trailing slash, e.g. https://x.up.railway.app)"

blank
check_health "$backend_url" || true
check_ready "$backend_url" "$shared_secret" || true
blank
note "A 401 above is worth stopping for: it means the secret in Railway is not"
note "the one this run holds, and every later step will inherit that."
pause "Press Enter to continue"

# --- 5. Vercel --------------------------------------------------------------

step "Step 5/8  Vercel — the frontend"
note "Import the same repository as a Vercel project and set:"
note "  Root directory     frontend   (the Next.js app is not at the repo root)"
note "  Framework preset   Next.js    — anything else builds cleanly and serves a"
note "                     site where every route 404s. docs/deployment.md → The"
note "                     Vercel trap."
blank
note "Environment variables — set BOTH of these for Production AND Preview:"
note "  CHAT_API_URL            $backend_url"
note "  BACKEND_SHARED_SECRET   the value from step 2, identical to Railway's"
blank
note "Previews talk to the same production backend with the same secret; there is"
note "no second backend. Variables set for Production only is the classic cause of"
note "'production works, every preview fails on every question'."
blank
note "Neither variable may ever be given a NEXT_PUBLIC_ prefix. Both are"
note "server-side only, and the prefix would ship the backend URL and the secret"
note "to the browser."
blank
note "Ignored build step — the command Vercel runs to decide whether to build."
note "Run from the project's root directory (frontend):"
blank
printf '        %s\n' 'git diff --quiet HEAD^ HEAD -- .'
blank
note "It exits 0 when nothing under frontend/ changed in the pushed commit, and"
note "Vercel skips the build on 0. A commit with no HEAD^ errors out and builds,"
note "which is the safe direction to fail."
blank
pause "Press Enter when the first Vercel deploy is live"

ask_url frontend_url "Frontend production URL (e.g. https://your-app.vercel.app)"

# --- 6. close the loop ------------------------------------------------------

step "Step 6/8  Back to Railway — close the loop"
note "The backend could not be told the frontend's origin until the frontend"
note "existed. Now it does. In Railway, set on the service:"
blank
printf '        ALLOWED_ORIGINS=%s\n' "$frontend_url"
blank
note "Comma-separated, no spaces, if you add the branch-preview origin too:"
printf '        %s,%s\n' "$frontend_url" "https://your-app-git-main-you.vercel.app"
blank
note "This is defence in depth and nothing more — no browser request ever reaches"
note "the backend, so CORS guards a path nothing uses. The shared secret is what"
note "protects the service. A stale value is still a misleading thing to leave."
note "Redeploy the backend afterwards: a variable change does not retroactively"
note "change a running deployment."
blank
pause "Press Enter when the backend has redeployed"

blank
note "Now the whole chain: browser → Vercel route → shared secret → Railway →"
note "retrieval → Groq → stream back. This asks the deployed site a real question."
note "Allow for a cold start; it may take a minute or two."
check_chat "$frontend_url" || true
blank
note "Worth doing by hand as well, on a backend that has been asleep: load the"
note "page, watch the status line read 'Nietzsche is stirring…' and go quiet, then"
note "ask something. A starter question clicked during the wake is held and"
note "dispatched when the backend reports ready — that is intended, not a hang."
pause "Press Enter to continue"

# --- 7. branch protection ---------------------------------------------------

step "Step 7/8  Branch protection on main"
note "Neither platform waits for CI: both react to the push itself, so a red"
note "build ships unless main is protected. In GitHub: Settings → Branches (or"
note "Rules) → protect main → require status checks to pass before merging."
blank
note "Branch protection matches on the check NAME. These run on every pull"
note "request to main, so they are safe to require — copy them verbatim:"
blank
printf '        %s\n' \
  'CodeQL (javascript-typescript)' \
  'CodeQL (python)' \
  'Secret scanning (gitleaks)' \
  'npm audit (frontend)' \
  'pip-audit (backend)' \
  'Dependency review'
blank
note "These others exist but are path-filtered — read the caveat before requiring"
note "any of them:"
blank
printf '        %s\n' \
  'Ruff lint & format' \
  'Pytest' \
  'Lint, format & types' \
  'Unit tests' \
  'Production build' \
  'Lighthouse CI'
blank
note "The caveat: backend-ci.yml, frontend-ci.yml and lighthouse.yml only run when"
note "their paths change, so a backend-only pull request never runs the frontend"
note "jobs. A required check that never runs stays pending forever and the pull"
note "request can never merge. Either require only the unfiltered checks above, or"
note "add a skip-job reporting the same name when the paths do not match."
note "The full list and what runs where: docs/ci-cd.md."
blank
if confirm "Have you configured branch protection?"; then
  protection_state="configured"
else
  protection_state="skipped"
  note "Left for later. Until then, CI is advisory and a red main still deploys."
fi

# --- 8. local env files -----------------------------------------------------

step "Step 8/8  Local development files — optional"
note "backend/.env and frontend/.env.local are what 'make dev' reads. Both are"
note "gitignored; neither is touched unless you ask here."
blank
note "The local files get their OWN secret, not the production one. Locally the"
note "value only has to match between the two files, and a production secret is"
note "worth less on a laptop than in Railway."
blank
if confirm "Create backend/.env and frontend/.env.local from the examples?"; then
  if ! local_secret="$(generate_secret)"; then
    result_warn "could not generate a local secret; skipping the env files"
  else
    for pair in "backend/.env:backend/.env.example" "frontend/.env.local:frontend/.env.example"; do
      target="$repo_root/${pair%%:*}"
      example="$repo_root/${pair##*:}"
      rel="${pair%%:*}"

      if ! (cd "$repo_root" && git check-ignore -q "$rel"); then
        result_fail "$rel is NOT gitignored — refusing to write a secret into it"
        continue
      fi
      if [ ! -f "$example" ]; then
        result_warn "${pair##*:} is missing; skipping $rel"
        continue
      fi
      if [ -e "$target" ]; then
        if ! confirm "$rel already exists. Overwrite it?"; then
          note "Left alone: $rel"
          continue
        fi
      fi

      while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
          BACKEND_SHARED_SECRET=*) printf 'BACKEND_SHARED_SECRET=%s\n' "$local_secret" ;;
          *) printf '%s\n' "$line" ;;
        esac
      done <"$example" >"$target"
      chmod 600 "$target"
      result_pass "wrote $rel (gitignored, mode 600)"
    done
    unset local_secret
    blank
    note "backend/.env still needs a real GROQ_API_KEY — nothing local works without"
    note "one. Leave SENTRY_DSN commented out; that is the point of it."
  fi
else
  note "Skipped. docs/deployment.md → Local environment has the two cp commands."
fi

# --- done -------------------------------------------------------------------

blank
say "================================================================="
say " Done"
say "================================================================="
say " Backend (Railway):   ${backend_url:-not set}"
say " Frontend (Vercel):   ${frontend_url:-not set}"
say " Sentry:              $sentry_state"
say " Branch protection:   $protection_state"
summarise || exit 1
blank
say "Re-check this deployment at any time, without prompts:"
say "  scripts/setup-deploy.sh --verify-only \\"
say "    --backend-url $backend_url --frontend-url $frontend_url"
say "  (with BACKEND_SHARED_SECRET exported, so readiness is checked too)"
blank
say "The checklist to tick off, and everything this wizard did not explain:"
say "  docs/deployment.md"
exit 0
