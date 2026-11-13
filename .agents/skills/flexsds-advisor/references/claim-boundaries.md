# Claim boundaries

- FACT: canonical v0.2 serving is linear adaptive speculative decoding. A class,
  banner, help text or UI labelled TreeAdaptive is not tree runtime evidence.
  TreeAdaptive runtime and arbitrary external tree execution remain future work.
- OBSERVATION: report detected hardware, selected profile, warning text and printed
  command with version/environment/exit code. A CPU fallback may hide a Torch or
  driver failure. Profile selection is lookup, not calibration or optimality.
- INFERENCE: compatibility is a name/VRAM heuristic. It neither validates weights,
  tokenizer alignment, quantization kernels nor proves memory fit. L4 fallback on
  other hardware cannot support measured hardware-aware policy claims.
- RECOMMENDATION: suggest a dry-run or a controlled benchmark; keep model changes
  explicit and explain assumptions. Do not invent alpha, acceptance, latency or
  percentage speedups. Analytical cost-model outputs are predictions, not measurements.
- UNKNOWN: adaptive-vs-vanilla speedup, profitability and no-slowdown behavior
  without controlled comparable measurements of the exact workload/configuration.
  Historical M4-02-04 evidence (including 49/49 acceptance) is configuration-specific
  and cannot generalize. Matching hardware alone is insufficient.
- IMPLEMENTED WITH LIMITATION: chat streams HTTP/SSE and monitor reads JSONL.
  Chunk-rate metrics are not token throughput or TPOT. Preflight blocks unsafe
  launch estimates; it does not prove model fit or cloud endpoint readiness.
- Deferred: performance diagnosis, refactoring external OpenAI clients, optimal
  policy selection, distributed serving, external publication and cross-agent
  support. No real cloud deployment is verified by this skill's local tests.

Do not repeat README/product guarantees as established results. A successful
preview does not prove launch, readiness, endpoint compatibility or performance.
