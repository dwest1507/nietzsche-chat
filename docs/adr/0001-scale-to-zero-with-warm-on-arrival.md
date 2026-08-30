# Backend scales to zero, warmed on arrival

The Railway backend holds ~1–1.5 GB resident permanently — two sentence-transformers
models plus the FAISS and BM25 indexes — because the whole point of the warm pipeline is
never to reload them. Railway bills resources held rather than requests served, so an
always-on container costs roughly $10–15/month, and this is the second such backend on
the account. That is out of proportion to a portfolio piece, so the service is configured
to **scale to zero** and the cold start is hidden instead of paid for: the frontend pings
the health endpoint when the chat mounts, so the container wakes and loads its models
while the visitor is still reading the page and typing.

## Consequences

- A visitor who clicks a starter question immediately can still outrun the warm-up. The
  frontend holds that message until the backend reports ready rather than sending it into
  a cold backend — see [0003](0003-readiness-is-separate-from-liveness.md).
- Keep-warm pings on a short interval (e.g. UptimeRobot every 5 minutes) are **not** a
  middle ground here: they prevent sleeping entirely, so they cost the same as always-on
  while adding a moving part. Don't add one thinking it's a compromise.
- If traffic ever becomes steady enough that the container rarely sleeps, always-on
  becomes the simpler choice and this decision should be revisited.
