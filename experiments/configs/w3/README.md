# Workload W3 - Free-form Vietnamese VAT/business analysis

- Workload ID: `w3_free_analysis`
- Dataset version: `w3.m1_03.v1`
- Prompt version: `w3_free_analysis.v1`
- Entropy profile: `high`
- Sample count: `12`
- Package SHA-256: `b06f32df357b5c60550f98817d9f6a1aade79b2258aafcb345e7b657cfd011b3`
- Prompt SHA-256: `44d81e7e753441bf918cbb48ace85b356081993e0d1a71016709ef37266d5637`
- Max tokens: `192`

Canonical files for M1-02/M1-03:

- `samples.jsonl`: model-facing prompt rows.
- `provenance.jsonl`: audit trail for the prompt source and row hashes.
- `chat_requests.jsonl`: replayable OpenAI-compatible chat requests.
- `package_manifest.json`: hashes and freeze metadata.

Do not rebuild prompts inside the benchmark harness; consume `chat_requests.jsonl`.
