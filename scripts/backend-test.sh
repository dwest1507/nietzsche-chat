#!/usr/bin/env bash
# Backend tests (pytest; ML models and Groq are mocked by the suite).
# Called by `make backend-test` and .github/workflows/backend-ci.yml.
set -euo pipefail
cd "$(dirname "$0")/../backend"

# Same dummy values the CI job uses; tests never call real services
export GROQ_API_KEY="${GROQ_API_KEY:-test_key}"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:3000}"
export BACKEND_SHARED_SECRET="${BACKEND_SHARED_SECRET:-test_shared_secret}"

echo "==> Pytest"
uv run pytest tests/ -v

echo "backend-test: OK"
