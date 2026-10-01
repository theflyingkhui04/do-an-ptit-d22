# Workload W2 - Text-to-SQL over Vietnamese VAT invoice schema

- Workload ID: `w2_text_to_sql`
- Dataset version: `w2.m1_03.v1`
- Prompt version: `w2_text_to_sql.v1`
- Entropy profile: `medium`
- Sample count: `12`
- Package SHA-256: `b2d92487cbe356ab9f7201865ed69fadab25c45fabf5f97fe5ef4c71f9da0cd6`
- Prompt SHA-256: `6eadba66bcb05d3469feafa2c93f2fa0e80a759ac390501b1410bd9037ffe36b`
- Max tokens: `96`

Canonical files for M1-02/M1-03:

- `samples.jsonl`: model-facing prompt rows.
- `provenance.jsonl`: audit trail for the prompt source and row hashes.
- `chat_requests.jsonl`: replayable OpenAI-compatible chat requests.
- `package_manifest.json`: hashes and freeze metadata.

Do not rebuild prompts inside the benchmark harness; consume `chat_requests.jsonl`.
