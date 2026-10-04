# Workload W2 v2 - Pipeline-aligned Vietnamese VAT Text-to-SQL over SQLite stable views

- Workload ID: `w2_text_to_sql`
- Dataset version: `w2.m1_03.v2`
- Prompt version: `w2_text_to_sql.v2`
- Entropy profile: `medium`
- Sample count: `12`
- Package SHA-256: `529f159c6837a9e2e326f70e0420eb80a1d0788f58d42a74734efee675fe7d9f`
- Prompt SHA-256: `40a0bff9f8a9701e992efeac38f60d48094bc5376649990ba5b22975e413605b`
- Max tokens: `96`
- Decision: `PM-approved W2-C pipeline-aligned diverse SQL + W3-D business-analysis-only diverse scenarios`

The frozen v1 package remains unchanged. This directory is a new v2 package.

Canonical files:

- `samples.jsonl`: model-facing prompt rows and scenario/query-pattern metadata.
- `provenance.jsonl`: decision/source audit trail and prompt hashes.
- `chat_requests.jsonl`: replayable OpenAI-compatible chat requests.
- `package_manifest.json`: hashes and freeze metadata.

Do not rebuild prompts inside the benchmark harness; consume `chat_requests.jsonl`.

W2 v2 rules:
- SQLite only; no PostgreSQL dialect request.
- Read-only stable views only: `v_invoice_facts`, `v_item_facts`, `v_tax_facts`.
- SQL-only benchmark response to keep W2 semi-structured/medium-entropy.
- Prompt pack covers filters, aggregates, GROUP BY, ORDER BY/TOP-K, HAVING, CTE/join, count-distinct, ratio, and multi-condition queries.
