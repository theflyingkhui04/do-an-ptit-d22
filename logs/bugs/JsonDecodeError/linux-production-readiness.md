# Linux production deployment appears stuck after App deployed

Local evidence: `~/.flexsds/logs/modal-serving-h_orwwsd.log` ends after a
successful 4.317-second deployment. This file contains publication logs only.

Read-only runtime inspection with the configured FlexSDS credentials and
`modal app logs flexsds-serving --tail 160 --timestamps` showed on 2026-11-17
(Asia/Bangkok):

- 09:17:07: vLLM API server starts.
- 09:17:51: model loading completes, 6.96 GiB.
- 09:19:13: JIT kernel warmup starts.
- 09:21:01: JIT warmup finishes after 107.37 seconds.
- 09:21:05: application startup complete; `/health` returns 200.

Subsequent live HTTP checks against the production endpoint succeeded for
`/health`, `/v1/models`, and `/v1/completions` with prompt `Hello`, max_tokens=1.
The served model ID is `qwen2.5-7b-adaptive`; completion returned one token.
This establishes that the deployed runtime can serve inference. It does not
prove the original local CLI completed its readiness loop.

The CLI captured only the `modal deploy` subprocess. Unlike `modal serve`, that
process exits after publication and does not stream subsequent GPU startup
logs. A cold start of about four minutes therefore looked idle in Ctrl+O.
This runtime evidence does not show the earlier Windows JSON decoding error.

Fix: wait for deployment to exit successfully before probing its URL, then
follow `modal app logs flexsds-serving --follow` during readiness. Runtime logs
feed the same progress display and Ctrl+O viewer. Stop the log follower on
success, error, or interruption without stopping the deployed application.
Expose readiness stages for health, model discovery, completion, and retries.

Regression tests cover successful/failed publication, runtime log collection,
and log follower cleanup on interruption. No new cloud deployment was made
to validate the code change; live inference checks used the existing app.

Modal documents on-demand web server startup:
https://modal.com/docs/guide/webhooks#non-asgi-web-servers
