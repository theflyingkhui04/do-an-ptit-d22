# Workload W3 v2 - Vietnamese VAT business-analysis free-form generation

- Workload ID: `w3_free_analysis`
- Dataset version: `w3.m1_03.v2`
- Prompt version: `w3_business_analysis.v2`
- Entropy profile: `high`
- Sample count: `12`
- Package SHA-256: `657ea19f30726372e08861f98b352ea7954db16cfcd165165a4db4a1be907689`
- Prompt SHA-256: `fee80fb2e4e1499f264ed8545337a0eda19a55c901c1ad461b32542a3fb85bd7`
- Max tokens: `768`
- Decision: `PM-approved W2-C pipeline-aligned diverse SQL + W3-D business-analysis-only diverse scenarios`

The frozen v1 package remains unchanged. This directory is a new v2 package.

Canonical files:

- `samples.jsonl`: model-facing prompt rows and scenario/query-pattern metadata.
- `provenance.jsonl`: decision/source audit trail and prompt hashes.
- `chat_requests.jsonl`: replayable OpenAI-compatible chat requests.
- `package_manifest.json`: hashes and freeze metadata.

Do not rebuild prompts inside the benchmark harness; consume `chat_requests.jsonl`.

W3 v2 rules:
- Business-analysis scenarios only; no thesis/speculative-decoding/meta-benchmark prompts.
- Diverse scenarios cover trends, concentration, product mix, anomaly/risk, compliance, data quality, reconciliation, audit prioritization, dashboards, and recommendations.
- v2 freezes `max_tokens=768`, PM-approved after diagnostics showed 12/12 truncation at 192, 5/12 truncation at 512, and a diagnostic-only 1024 probe observed natural completions up to 681 tokens.
