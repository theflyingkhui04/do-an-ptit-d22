# M1 Workload Prompt Sets

This directory stores committed prompt sets for M1 baseline benchmark sweeps.

- `w2_text_to_sql.jsonl`: medium-entropy Vietnamese Text-to-SQL prompts over a VAT invoice schema.
- `w3_free_analysis.jsonl`: high-entropy Vietnamese free-form VAT analysis prompts.
- `w1_vat_extractor.jsonl`: low-entropy VAT JSON extraction prompts.

Each JSONL row has:

- `id`: stable prompt id.
- `workload`: canonical workload name.
- `prompt`: full user prompt sent to `/v1/chat/completions`.

The benchmark harness cycles through prompt rows by request index, so the same file can be used for quick smoke runs and larger concurrency sweeps.
