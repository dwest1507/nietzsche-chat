# Readiness is a separate, authenticated endpoint from liveness

The backend binds its port immediately and loads the retrieval models on a background
thread, so "the process is up" and "the process can answer a question" are genuinely
different states for tens of seconds after every wake. They are exposed as two endpoints:

- **`/api/health`** — open, flat liveness. This is what Railway probes. It reports nothing
  about the models, deliberately: a healthcheck that waited for readiness would block
  every deploy on the full model load, which is the stall the background thread exists to
  avoid.
- **`/api/ready`** — reports pipeline state, and requires the shared secret from
  [0002](0002-shared-secret-gateway.md). The frontend polls it through a Next route while
  a starter question is held, and only sends the message once the pipeline is ready.

Neither endpoint may call `get_pipeline()`. That function blocks on the singleton lock
while the warm thread loads, so probing through it would make both endpoints hang for the
exact duration they exist to report on. Readiness must be read from a flag the warm thread
sets.

## Consequences

- `/api/health` has to stay unauthenticated for Railway to probe it, which makes it a
  public wake-up button for a service that scales to zero
  ([0001](0001-scale-to-zero-with-warm-on-arrival.md)). Someone looping it can hold the
  container awake at our expense; the sleep timeout caps the damage.
- A deploy can go green while the container still cannot answer a question. That is
  intended — the frontend, not Railway, is what waits for readiness.
