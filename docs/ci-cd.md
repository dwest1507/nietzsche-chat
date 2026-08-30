# CI/CD

Four workflows in `.github/workflows/`. All of the check logic lives in `scripts/`, which
is what `make ci-cd` runs, so a green local run means a green CI run. Nothing here
deploys: Railway and Vercel deploy from `main` through their own git integrations, and
Actions holds no deploy tokens. See [deployment.md](deployment.md).

## Required status checks

Branch protection matches on the check name, which is the job's `name:`. Copy these
verbatim.

Always run on a pull request to `main` — safe to require:

```
CodeQL (javascript-typescript)
CodeQL (python)
Secret scanning (gitleaks)
npm audit (frontend)
pip-audit (backend)
Dependency review
```

Path-filtered — see the caveat below before requiring any of these:

```
Ruff lint & format
Pytest
Lint, format & types
Unit tests
Production build
Lighthouse CI
```

| Check                            | Workflow          | Runs on                                                                     |
| -------------------------------- | ----------------- | --------------------------------------------------------------------------- |
| `Ruff lint & format`             | `backend-ci.yml`  | changes under `backend/**`, `scripts/backend-*.sh`, the workflow            |
| `Pytest`                         | `backend-ci.yml`  | same                                                                        |
| `Lint, format & types`           | `frontend-ci.yml` | changes under `frontend/**`, `scripts/frontend-*.sh`, the workflow          |
| `Unit tests`                     | `frontend-ci.yml` | same                                                                        |
| `Production build`               | `frontend-ci.yml` | same                                                                        |
| `Lighthouse CI`                  | `lighthouse.yml`  | pull requests touching `frontend/**`, `scripts/lighthouse.sh`, the workflow |
| `CodeQL (javascript-typescript)` | `security.yml`    | every pull request to `main`, every push to `main`, weekly                  |
| `CodeQL (python)`                | `security.yml`    | same                                                                        |
| `Secret scanning (gitleaks)`     | `security.yml`    | same                                                                        |
| `npm audit (frontend)`           | `security.yml`    | same                                                                        |
| `pip-audit (backend)`            | `security.yml`    | same                                                                        |
| `Dependency review`              | `security.yml`    | pull requests only                                                          |

**The path-filter caveat.** `backend-ci.yml`, `frontend-ci.yml` and `lighthouse.yml` are
path-filtered, so a backend-only pull request never runs the frontend jobs and vice versa.
A required check that never runs is reported as pending forever, and the pull request
cannot be merged. Either require only the unfiltered `security.yml` checks, or add a
skip-job that reports the same name when the paths do not match.

## What runs where

| Job                   | Script                        | Also as                 |
| --------------------- | ----------------------------- | ----------------------- |
| Backend lint          | `scripts/backend-lint.sh`     | `make backend-lint`     |
| Backend tests         | `scripts/backend-test.sh`     | `make backend-test`     |
| Frontend quality      | `scripts/frontend-quality.sh` | `make frontend-quality` |
| Frontend tests        | `scripts/frontend-test.sh`    | `make frontend-test`    |
| Frontend build        | `scripts/frontend-build.sh`   | `make frontend-build`   |
| Lighthouse            | `scripts/lighthouse.sh`       | `make lighthouse`       |
| npm audit / pip-audit | `scripts/security-audit.sh`   | `make security-audit`   |

CodeQL, gitleaks and dependency review are GitHub-hosted analyses with no local
equivalent.

## Secrets

CI needs none. The backend suite mocks the RAG pipeline and the Groq client and runs on
dummy values exported by `scripts/backend-test.sh`; `SENTRY_DSN` is exported empty there
so the suite cannot report its own deliberate failures anywhere. The only token in use is
the `GITHUB_TOKEN` that Actions provides to gitleaks.

## CI does not gate deploys

Both platforms react to the push, not to the check results, so a red build still ships
unless `main` is branch-protected with the checks above.
